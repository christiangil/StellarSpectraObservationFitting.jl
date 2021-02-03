## Setup
cd(@__DIR__)

# include("setup.jl")
# using RvSpectML
# RSM = RvSpectML
using NPZ
using UnitfulAstro, Unitful
using HDF5
using Statistics
using Zygote
using TemporalGPs
using Plots
using Optim
using Flux

## Getting precomputed airmasses and observation times
valid_obs =
    npzread("airmasses/valid_obs_res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.npy")
valid_inds = [i for i in 1:length(valid_obs) if valid_obs[i]]
times_nu =
    npzread("airmasses/times_res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.npy")[valid_inds]
times = times_nu .* 1u"d"
n_obs = length(times)
airmasses =
    npzread("airmasses/airmasses_res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.npy")[valid_inds]

include("../src/general_functions.jl")
include("../src/PCA_functions.jl")
include("../src/SOAP_functions.jl")
include("D:/Christian/Documents/GitHub/GLOM_RV_Example/src/GLOM_RV_Example.jl")
GLOM_RV = Main.GLOM_RV_Example

stellar_activity = true
incl_tellurics = true
bary_velocity = true
refit_SOAP_GP = false
SNR = 1000
K = 10u"m/s"
# T = 5700u"K"
min_wav = 615  # nm
max_wav = 665  # nm
obs_resolution = 150000
star_template_res = obs_resolution * sqrt(2)  # change to center pixels on median pixel?
SOAP_gp_ridge = 1e-14

## Simulating observations

fid =
    h5open("D:/Christian/Downloads/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.h5")

quiet = fid["quiet"][:] # .* planck.(λ, T)
nz_inds = quiet .!= 0
nz_inds = findfirst(nz_inds):(findlast(nz_inds)-1)
# nz_inds = [i for i in 1:length(quiet) if quiet[i]!=0]
quiet = quiet[nz_inds]
λ = fid["lambdas"][nz_inds]u"nm" / 10

if stellar_activity
    ys = fid["active"][nz_inds, :] #.* planck.(λ, T)
else
    ys = repeat(quiet, 1, 730)
end
close(fid)

ys ./= maximum(ys)
quiet ./= maximum(quiet)

λ_nu = ustrip.(λ)

# integrated_first = trapz(λ_nu, ys[:, 1])
# for i in 1:size(ys, 2)
#     ys[:, i] *= integrated_first / trapz(λ_nu, ys[:, i])
# end
# quiet*= integrated_first / trapz(ustrip.(λ), quiet)

active = ys[:, valid_inds]

make_template(matrix::Matrix{T}) where {T<:Real} = vec(median(matrix, dims=2))
# dont use mean for initialization because variance of one type of activity is
# attributed to the other
make_template_mean(matrix::Matrix{T}) where {T<:Real} =
    vec(mean(matrix, dims = 2))

function DPCA(
    spectra::Matrix{T},
    λs::Vector{T};
    template::Vector{T} = make_template(spectra),
    num_components::Int = 3,
) where {T<:Real}
    doppler_comp = calc_doppler_component_RVSKL(λs, spectra)
    return fit_gen_pca_rv_RVSKL(
        spectra,
        doppler_comp,
        mu = template,
        num_components = num_components,
    )
end

x, x, x, x, rvs_activ_no_noise = DPCA(active, λ_nu)

## Fitting GP to SOAP spectra

function build_gp(θ)
    σ², l = θ
    k = σ² * TemporalGPs.stretch(TemporalGPs.Matern52(), 1 / l)
    f_naive = TemporalGPs.GP(k, TemporalGPs.GPC())
    f = to_sde(f_naive, SArrayStorage(Float64))
    #f = to_sde(f_naive)   # if develop issues with StaticArrays could revert to this
end

# θ1 = [0.029636118846587914, 1.2930739377171542e-5]  # -ℓ = -3.526166289873243e6
θ1 = [3.3270754364467443, 9.021560480866474e-5]  # -ℓ = -3.5697490959271505e6
train_y = quiet .- 1
log_λ_nu = log.(λ_nu)

# refit_SOAP_GP = true
# refit_SOAP_GP = false
if refit_SOAP_GP
    function NLL(θ)
        gp = build_gp(θ)
        gpx = gp(log_λ_nu, SOAP_gp_ridge)
        return -logpdf(gpx, train_y)
    end

    f(θ) = NLL(exp.(θ))
    function g!(G, θ)
        G[:] = gradient(() -> f(θ), Params([θ]))[θ]
    end

    # ends optimization if true
    function optim_cb(x::OptimizationState)
        println()
        if x.iteration > 0
            println("Iteration:       ", x.iteration)
            println("Time so far:     ", x.metadata["time"], " s")
            println("-Log Likelihood: ", x.value)
            println("Gradient norm:   ", x.g_norm)
            println()
        end
        return false
    end

    θ_guess = log.([5e-1, 1e-6])  # log([line height, line width in log space]
    result = optimize(
        f,
        g!,
        θ_guess,
        LBFGS(; alphaguess = 1e-6),
        Optim.Options(callback=optim_cb, g_tol=1e-1, iterations=15),
    ) # 300s, 5 iters

    θ1[:] = exp.(result.minimizer)
end
println(θ1)

SOAP_gp = build_gp(θ1)
gpx = SOAP_gp(log_λ_nu, SOAP_gp_ridge)
gp_post = posterior(gpx, train_y)
gpx_post = gp_post(log_λ_nu)

post_dist = TemporalGPs.marginals(gpx_post)
pred_y = mean.(post_dist) .+ 1
std_y = std.(post_dist)

plot_spectrum(; kwargs...) = plot(;
    xlabel = "Wavelength (nm)",
    ylabel = "Continuum Normalized Flux",
    dpi = 400,
    kwargs...,
)
plot_rv(; kwargs...) =
    plot(; xlabel = "Time (d)", ylabel = "RV (m/s)", dpi = 400, kwargs...)

predict_plot = plot_spectrum(; xlim = (450.80, 450.85))
plot!(
    predict_plot,
    λ_nu,
    pred_y,
    ribbons = std_y,
    st = :line,
    fillalpha = 0.3,
    lw = 2,
    color = :blue,
    label = "GP",
)
plot!(
    predict_plot,
    λ_nu,
    train_y .+ 1,
    st = :scatter,
    color = :red,
    ms = 3,
    msw = 0.5,
    label = "SOAP",
)
png(predict_plot, "figs/test1.png")

predict_plot = plot_spectrum(; legend = :bottomright)
plot!(
    predict_plot,
    λ_nu,
    pred_y,
    ribbons = std_y,
    st = :line,
    fillalpha = 0.3,
    lw = 1,
    color = :blue,
    label = "GP",
)
plot!(
    predict_plot,
    λ_nu,
    train_y .+ 1,
    st = :line,
    color = :red,
    lw = 1,
    label = "SOAP",
    alpha = 0.5,
)
png(predict_plot, "figs/test2.png")

## Reading in O2 and H2O lines from HITRAN

fid = open("lines/outputtransitionsdata3.par", "r")
n_lines = countlines(open("lines/outputtransitionsdata3.par", "r"))
is_h2o = [false for i = 1:n_lines]
is_o2 = [false for i = 1:n_lines]
intensities = zeros(n_lines)
wavelengths = zeros(n_lines)
for i = 1:n_lines
    new_line = readline(fid)
    if length(new_line) > 0
        is_h2o[i] = new_line[2] == '1'
        is_o2[i] = new_line[2] == '7'
        intensities[i] = parse(Float64, new_line[17:25])
        wavelengths[i] = 1e7 / parse(Float64, new_line[4:15])
    end
end
close(fid)

intens_h2o = intensities[is_h2o]
wavelengths_h2o = wavelengths[is_h2o]
max_intens_h2o_inds = sortperm(intens_h2o)[end:-1:end-9]

intens_o2 = intensities[is_o2]
wavelengths_o2 = wavelengths[is_o2]
max_intens_o2_inds = sortperm(intens_o2)[end:-1:end-1]

## Creating telluric mask

inds = [λ1 > min_wav && λ1 < max_wav for λ1 in λ_nu]

# SOAP_resolution =
#     sqrt(λ_nu[end] * λ_nu[1]) * length(λ_nu) / (λ_nu[end] - λ_nu[1])
# psf_width = 6 / SOAP_resolution  # in units of 1/R
# 4 for half width of NEID, 8-12 for half width of EXPRESS?

# find a line in SOAP
line_min = 664.3
line_max = 664.4
line_center = sqrt(line_min * line_max)
inds2 = [λ1 > line_min && λ1 < line_max for λ1 in λ_nu]
# inds2 = [λ1 > 635.8 && λ1 < 635.9 for λ1 in λ_nu]
# plot(λ[inds2], quiet[inds2])
quiet_line = quiet[inds2]
quiet_λ = λ_nu[inds2]
maximum(quiet_line)
HM = (maximum(quiet_line) + minimum(quiet_line)) / 2

line_peak = quiet_λ[quiet_line.<HM]
FWHM = line_peak[end] - line_peak[1]
solar_sigma = FWHM / (2 * sqrt(2 * log(2)))
telluric_sigma = solar_sigma / sqrt(2) / line_center

n_mask =
    Int(round((max_wav - min_wav) * obs_resolution / sqrt(max_wav * min_wav)))
obs_λ = log_linspace(min_wav, max_wav, n_mask)

gauss(x::Real; a::Real = 1, loc::Real = 0, sigma::Real = 1) =
    a * exp(-((x - loc) / sigma)^2 / 2)

global h2omask = ones(n_mask)

for wave in wavelengths_h2o[max_intens_h2o_inds]
    global h2omask .*=
        1 .-
        gauss.(
            obs_λ,
            a = 0.3 + 0.5 * rand(),
            loc = wave,
            sigma = telluric_sigma * wave,
        )
end
wavelengths_h2o[max_intens_h2o_inds]

global o2mask = ones(n_mask)

for wave in wavelengths_o2[max_intens_o2_inds]
    o2mask .*=
        1 .-
        gauss.(
            obs_λ,
            a = 0.2 + 0.6 * rand(),
            loc = wave,
            sigma = telluric_sigma * wave,
        )
end

function tellurics(airmass::Real; add_rand::Bool = true)
    @assert 1 <= airmass <= 11.13
    o2scale = airmass / 2
    h2oscale = airmass / 2 * (0.7 + 0.6 * (rand() - 0.5) * add_rand)
    return (o2mask .^ o2scale) .* (h2omask .^ h2oscale)
end

## Bringing observations into observer frame and multiplying by telluric mask
# aka finishing simulating the data
bary_K = sqrt(1.32712440042e20u"m^3/s^2" / 1.496e+11u"m")
planet_P_nu = sqrt(700)
planet_ks = GLOM_RV.kep_signal(K = K, e_or_h = 0.1, P = planet_P_nu * u"d")
bary_ks = GLOM_RV.kep_signal(
    K = bary_K / sqrt(2),
    e_or_h = 0.016,
    P = 1u"yr",
    M0 = rand() * 2π,
    ω_or_k = rand() * 2π,
)
rvs_kep = planet_ks.(times)
rvs_bary = Int(bary_velocity) .* bary_ks.(times)
vs = rvs_kep + rvs_bary
log_obs_λ = log.(obs_λ)

function get_marginal_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::Vector{T},
    xs::Vector{T},
) where {T<:Real}
    gp_post = posterior(finite_GP, ys)
    gpx_post = gp_post(xs)
    return TemporalGPs.marginals(gpx_post)
end

function get_mean_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::Vector{T},
    xs::Vector{T},
) where {T<:Real}
    return mean.(get_marginal_GP(finite_GP, ys, xs))
end

struct SpectraHolder{T<:AbstractArray{<:Real,1}}
    log_λ_obs::T
    log_λ_bary::T
    flux_obs::T
    var_obs::T
    function SpectraHolder(
        log_λ_obs::T,
        log_λ_bary::T,
        flux_obs::T,
        var_obs::T,
    ) where {T<:AbstractArray{<:Real,1}}
        @assert 1 <=
                length(log_λ_obs) ==
                length(log_λ_bary) ==
                length(flux_obs) ==
                length(var_obs)
        new{typeof(log_λ_obs)}(log_λ_obs, log_λ_bary, flux_obs, var_obs)
    end
end

function shift_log_λ(v::Unitful.Velocity, log_λ::Vector{T}) where {T<:Real}
    return log_λ .+ (log((1.0 + v / light_speed) / (1.0 - v / light_speed)) / 2)
end

Spectra = Vector{SpectraHolder{Vector{Float64}}}()
true_tels = ones(length(obs_λ), n_obs)
incl_tellurics_int = Int(incl_tellurics)
normalization = quantile(quiet[inds], 0.9)
active_noisy = zero(true_tels)
active[:, :] .-= 1
@time for i = 1:n_obs  # 90s
    true_tels[:, i] =
        (1 - incl_tellurics_int) .+
        (incl_tellurics_int .* tellurics(airmasses[i]))

    log_true_λ = shift_log_λ(vs[i], log_obs_λ)
    log_bary_λ = shift_log_λ(rvs_bary[i], log_obs_λ)

    active_noisy[:, i] = get_mean_GP(gpx, active[:, i], log_obs_λ) .+ 1
    flux_obs =
        (get_mean_GP(gpx, active[:, i], log_true_λ) .+ 1) .*
        true_tels[:, i]

    active_noisy[:, i] /= normalization
    flux_obs /= normalization
    var_obs = flux_obs .* spectra_noise_ratios(flux_obs, obs_λ, SNR)
    noises = var_obs .* randn(n_mask)
    active_noisy[:, i] += noises
    flux_obs += noises
    var_obs .*= var_obs

    append!(Spectra, [SpectraHolder(log_obs_λ, log_bary_λ, flux_obs, var_obs)])
end
active[:, :] .+= 1
quiet /= normalization
true_tels_mean = vec(mean(true_tels, dims=2))

x, x, x, x, rvs_activ_noisy = DPCA(active_noisy, obs_λ)

using JLD2
@save "E:/telfitting/telfitting_workspace.jld2" FWHM HM K SNR SOAP_gp SOAP_gp_ridge Spectra active active_noisy airmasses bary_K bary_ks bary_velocity gp_post gpx gpx_post h2omask incl_tellurics incl_tellurics_int inds inds2 intens_h2o intens_o2 intensities is_h2o is_o2 light_speed light_speed_nu line_center line_max line_min line_peak log_obs_λ log_λ_nu max_intens_h2o_inds max_intens_o2_inds max_wav min_wav n_lines n_mask n_obs normalization nz_inds o2mask obs_resolution obs_λ planet_P_nu planet_ks post_dist pred_y predict_plot quiet quiet_line quiet_λ refit_SOAP_GP rvs_activ_no_noise rvs_activ_noisy rvs_bary rvs_kep solar_sigma star_template_res std_y stellar_activity telluric_sigma times times_nu train_y true_tels true_tels_mean valid_inds valid_obs vs wavelengths wavelengths_h2o wavelengths_o2 x ys θ1 λ λ_nu
@save "E:/telfitting/telfitting_workspace_smol.jld2" SOAP_gp SOAP_gp_ridge Spectra airmasses bary_K inds light_speed light_speed_nu max_wav min_wav n_obs obs_resolution obs_λ planet_P_nu planet_ks quiet rvs_activ_no_noise rvs_activ_noisy rvs_kep times times_nu true_tels true_tels_mean vs wavelengths θ1 λ λ_nu
# SOAP_gp_slow = build_gp_base(θ1)

# SOAP_gp_slow_bary_temp = SOAP_gp_slow(log_λ_star_template)
# Σ_bary = cov(SOAP_gp_slow_bary_temp, SOAP_gp_slow_bary_temp)
# Σ_bary[diagind(Σ_bary)] .+= 1e-4
# Σ_bary = cholesky(Σ_bary)
# @save "E:/telfitting/Sigma_bary.jld2" Σ_bary

# Σ_obs_bary = ones((len_obs, len_bary))
# @time for i in 1:n_obs
#     Σ_obs_bary[:, :] = cov(SOAP_gp_slow(Spectra[i].log_λ_bary), SOAP_gp_slow_bary_temp)
#     @save "E:/telfitting/Sigma_obs_bary_$i.jld2" Σ_obs_bary
# end
