## Setup
cd(@__DIR__)

# include("setup.jl")
# using RvSpectML
# RSM = RvSpectML
using NPZ
using UnitfulAstro, Unitful
using HDF5
using Statistics
using Zygote  # Slow
using TemporalGPs
using Plots
using Optim
using Flux
using JLD2
using Stheno
using FileIO

## Getting precomputed airmasses and observation times

include("../src/general_functions.jl")
include("../src/PCA_functions.jl")
include("../src/SOAP_functions.jl")
include("D:/Christian/Documents/GitHub/GLOM_RV_Example/src/GLOM_RV_Example.jl")
GLOM_RV = Main.GLOM_RV_Example

make_template(matrix::Matrix{T}) where {T<:Real} = vec(median(matrix, dims=2))
make_template_mean(matrix::Matrix{T}) where {T<:Real} =
    vec(mean(matrix, dims = 2))

function DPCA(
    spectra::Matrix{T},
    λs::Vector{T};
    template::Vector{T} = make_template(spectra),
    num_components::Int = 3,
) where {T<:Real}
    doppler_comp = calc_doppler_component_RVSKL(λs, template)

    return fit_gen_pca_rv_RVSKL(
        spectra,
        doppler_comp,
        mu = template,
        num_components = num_components,
    )
end

function build_gp_base(θ)
    σ², l = θ
    k = σ² * TemporalGPs.stretch(TemporalGPs.Matern52(), 1 / l)
    return TemporalGPs.GP(k, TemporalGPs.GPC())
end

build_gp(θ) = to_sde(build_gp_base(θ), SArrayStorage(Float64))
# build_gp(θ) = to_sde(build_gp_base(θ))

plot_spectrum(; kwargs...) = plot(; xlabel = "Wavelength (nm)", ylabel = "Continuum Normalized Flux", dpi = 400, kwargs...)
plot_rv(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "RV (m/s)", dpi = 400, kwargs...)

function get_marginal_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::AbstractVector,
    xs::AbstractVector)
    gp_post = posterior(finite_GP, ys)
    gpx_post = gp_post(xs)
    return TemporalGPs.marginals(gpx_post)
end

function get_mean_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::AbstractVector,
    xs::AbstractVector)
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

function shift_λ(v::Unitful.Velocity, log_λ::Vector{T}) where {T<:Real}
    return log_λ .+ (log((1.0 + v / light_speed) / (1.0 - v / light_speed)) / 2)
end

@load "E:/telfitting/telfitting_workspace.jld2"

## 1) Estimating stellar template
max_bary_z = 2.5 * bary_K / light_speed
min_temp_wav = (1 - max_bary_z) * min_wav
max_temp_wav = (1 + max_bary_z) * max_wav
λ_star_template = log_linspace(min_temp_wav, max_temp_wav,
    Int(round((max_temp_wav - min_temp_wav) * star_template_res / sqrt(max_temp_wav * min_temp_wav))))
log_λ_star_template = log.(λ_star_template)
len_obs = length(obs_λ)
len_bary = length(log_λ_star_template)
telluric_obs = ones(len_obs, n_obs)  # part of model
flux_bary = ones(len_bary, n_obs)  # interpolation / telluric_obs
var_bary = ones(len_bary, n_obs)  # interpolation / telluric_obs
flux_obs = ones(len_obs, n_obs)
for i = 1:n_obs # 13s
    flux_obs[:, i] = Spectra[i].flux_obs
end
# flux_star  # part of model = make_template(flux_bary) + DPCA stuff

# SOAP_gp_slow = build_gp_base(θ1)
# SOAP_gp_slow_bary_temp = SOAP_gp_slow(log_λ_star_template)
# @load "E:/telfitting/Sigma_bary.jld2" Σ_bary

function flux_bary_gp(index::Int, telluric::Vector{T}) where {T<:Real}
    return get_marginal_GP(
        SOAP_gp(Spectra[index].log_λ_bary, Spectra[index].var_obs ./ telluric ./ telluric),
        (Spectra[index].flux_obs ./ telluric) .- 1,
        log_λ_star_template,
    )
end

function est_flux_bary!(
    fluxes::Matrix{T},
    vars::Matrix{T},
    tellurics::Matrix{T},
) where {T<:Real}
    for i = 1:size(fluxes, 2) # 15s
        gp = flux_bary_gp(i, tellurics[:, i])
        fluxes[:, i] = mean.(gp) .+ 1
        vars[:, i] = var.(gp)
    end
end
est_flux_bary!(flux_bary, var_bary, telluric_obs)
flux_star_template = make_template(flux_bary)
_, _, _, _, rvs_naive = DPCA(flux_bary, λ_star_template; template=flux_star_template, num_components=1)

predict_plot = plot_spectrum(; legend = :bottomright, size=(1200, 800))
# predict_plot = plot_spectrum(; xlim = (627.8, 628.3), legend = :bottomright)
plot!(predict_plot, λ_star_template, flux_star_template, st=:line, lw=1, label="Predicted Quiet")
plot!(predict_plot, λ_nu[inds], quiet[inds], st=:line, lw=1, label="SOAP Quiet", alpha = 0.5)
plot!(predict_plot, obs_λ, true_tels_mean, st=:line, lw=2, label="average telluric", alpha = 0.5)
png(predict_plot, "figs/test3.png")

# Compare first guess at RVs to true signal
plot_times = linspace(times_nu[1], times_nu[end], 1000)
plot_rvs_kep = ustrip.(planet_ks.(plot_times .* 1u"d"))
predict_plot = plot_rv()
plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Truth")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="First Guess")
png(predict_plot, "figs/test4.png")

# Compare RV differences to actual RVs from activity
rvs_kep_nu = ustrip.(rvs_kep)
predict_plot = plot_rv()
plot!(predict_plot, times_nu, rvs_naive - rvs_kep_nu, st=:scatter, ms=3, color=:blue, label="First Guess - Truth")
plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Actvity (with obs. SNR and resolution)")
png(predict_plot, "figs/test5.png")

## 2) Use estimated stellar template to estimate tellurics

function est_tellurics!(
    tellurics::Matrix{T},
    fluxes::Matrix{T},
    vars::Matrix{T},
) where {T<:Real}
    for i = 1:n_obs # 13s
        tellurics[:, i] = Spectra[i].flux_obs ./ (get_mean_GP(
            SOAP_gp(log_λ_star_template, vars[:, i]),
            fluxes[:, i] .- 1,
            Spectra[i].log_λ_bary) .+ 1)
    end
end

est_tellurics!(telluric_obs, repeat(flux_star_template, 1, n_obs), ones(size(flux_bary)) * SOAP_gp_ridge)
flux_telluric_guess = make_template(telluric_obs)
_, M_tel, s_tel, _ = fit_gen_pca(telluric_obs; num_components=2, mu=flux_telluric_guess)

## telluric sanity check plots

# Compare telluric guess to actual tellurics
predict_plot = plot_spectrum(; legend = :bottomright, size=(1200, 800))
# zoom in on o2 lines
# predict_plot = plot_spectrum(; xlim=(627.8,628.3), legend=:bottomright)
# zoom in on h2o lines
# predict_plot = plot_spectrum(; xlim=(651.5,652), legend=:bottomright)
plot!(predict_plot, obs_λ, true_tels_mean / norm(true_tels_mean), st=:line, lw=1.5, label="Average Telluric")
plot!(predict_plot, obs_λ, flux_telluric_guess / norm(flux_telluric_guess), st=:line, lw=1, label="Telluric guess mean", alpha=0.8)
# plot!(predict_plot, obs_λ, M_tel[:,1], st=:line, lw=1, label="Telluric guess component 1", alpha=0.5)
# plot!(predict_plot, obs_λ, M_tel[:,2], st=:line, lw=1, label="Telluric guess component 2", alpha=0.5)
png(predict_plot, "figs/test6.png")

## 3) Use estimated tellurics template to make first pass at improving RVs

est_flux_bary!(flux_bary, var_bary, repeat(flux_telluric_guess, 1, n_obs))
# est_flux_bary!(flux_bary, var_bary, telluric_obs .+ M_tel * s_tel)

flux_star_template = make_template(flux_bary)
_, M_star, s_star, _, rvs_notel = DPCA(flux_bary, λ_star_template; template = flux_star_template)

# Compare second guess at RVs to true signal
predict_plot = plot_rv()
plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Truth")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="First Guess")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel, st=:scatter, ms=3, color=:green, label="Second Guess")
png(predict_plot, "figs/test7.png")

# Compare RV differences to actual RVs from activity
predict_plot = plot_rv()
plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Actvity (with obs. SNR and resolution)")
plot!(predict_plot, times_nu, rvs_notel - rvs_kep_nu, st=:scatter, ms=3, color=:green, label="Second Guess - Truth")
png(predict_plot, "figs/test8.png")

std((rvs_naive - rvs_kep_nu) - rvs_activ_no_noise)
std((rvs_notel - rvs_kep_nu) - rvs_activ_no_noise)
std(rvs_activ_noisy - rvs_activ_no_noise)  # best case
std(rvs_notel - rvs_kep_nu)
std(rvs_activ_noisy)
std(rvs_activ_no_noise)

## What do tellurics at each time look like?

est_tellurics!(telluric_obs, repeat(flux_star_template, 1, n_obs), var_bary)
flux_telluric_guess = make_template(telluric_obs)
_, M_tel, s_tel, _ = fit_gen_pca(telluric_obs; num_components=2, mu=flux_telluric_guess)

plot_epoch = 60
predict_plot = plot_spectrum(; legend = :bottomright, size=(1200, 800))
# zoom in on o2 lines
# predict_plot = plot_spectrum(; xlim=(627.8,628.3), legend=:bottomright)
# zoom in on h2o lines
# predict_plot = plot_spectrum(; xlim=(651.5,652), legend=:bottomright, size=(1200,800))
plot!(predict_plot, obs_λ, true_tels[:, plot_epoch], #- flux_telluric_guess,
    st=:line, lw=2, label="actual tels - telluric template")
plot!(predict_plot, obs_λ, telluric_obs[:, plot_epoch], #- flux_telluric_guess,
    st=:line, lw=1.5, label="obs / stellar - telluric template", alpha=0.8)
plot!(predict_plot, obs_λ, (M_tel*s_tel)[:, plot_epoch] + flux_telluric_guess, st=:line, lw=1, label="basis * scores", alpha=0.8)


## fit star

obs_var = zeros(size(telluric_obs))
for i = 1:size(obs_var, 2) # 15s
    obs_var[:, i] = Spectra[i].var_obs
end

L1(thing) = sum(abs.(thing))
L2(thing) = sum(thing .* thing)

function status_plot()
    l = @layout [a; b]
    # predict_plot = plot_spectrum(; legend = :bottomleft, size=(800,1200), layout = l)
    # predict_plot = plot_spectrum(; xlim=(627.8,628.3), legend=:bottomleft, size=(800,1200), layout = l) # o2
    # predict_plot = plot_spectrum(; xlim=(651.5,652), legend=:bottomleft, size=(800,1200), layout = l)  # h2o
    predict_plot = plot_spectrum(; xlim = (647, 656), legend = :bottomleft, size=(800,1200), layout = l)  # h2o
    plot!(predict_plot[1], obs_λ, true_tels[:, plot_epoch], label="true tel")
    plot!(predict_plot[1], obs_λ, telluric_obs[:, plot_epoch], label="predicted tel", alpha = 0.5)
    tel_model!(tel_model_result, view(θ_holder, 1:3))
    plot!(predict_plot[1], obs_λ, tel_model_result[:, plot_epoch], label="model tel: $tracker", alpha = 0.5)
    plot!(predict_plot[2], exp.(Spectra[plot_epoch].log_λ_bary), Spectra[plot_epoch].flux_obs ./ true_tels[:, plot_epoch], label="obs / true tel", )
    star_model!(star_model_bary, star_model_result, M_rv, s_rv, view(θ_holder, 4:6))
    plot!(predict_plot[2], λ_star_template, flux_bary[:, plot_epoch], label="predicted star", alpha = 0.5)
    plot!(predict_plot[2], λ_star_template, star_model_bary[:, plot_epoch] .+ 1, label="model star: $tracker", alpha = 0.5)
    display(predict_plot)
end


function reset_fit!()
    telluric_obs[:, :] .= 1  # part of model
    flux_bary[:, :] .= 1  # interpolation / telluric_obs
    var_bary[:, :] .= 1

    # model everything as stellar variability
    est_flux_bary!(flux_bary, var_bary, telluric_obs)
    flux_star_template[:] = make_template(flux_bary)
    _, M_star[:, :], s_star[:, :], _, rvs_notel[:] =
        DPCA(flux_bary, λ_star_template; template = flux_star_template)


    # telluric model with stellar template
    est_tellurics!(
        telluric_obs,
        repeat(flux_star_template, 1, n_obs),
        # ones(size(flux_bary)) * SOAP_gp_ridge,
        var_bary,
    )
    flux_telluric_guess[:] = make_template(telluric_obs)
    _, M_tel[:, :], s_tel[:, :], _ =
        fit_gen_pca(telluric_obs; num_components=2, mu=flux_telluric_guess)

    # stellar model with telluric template
    est_flux_bary!(flux_bary, var_bary, repeat(flux_telluric_guess, 1, n_obs))
    flux_star_template[:] = make_template(flux_bary)
    _, M_star[:, :], s_star[:, :], _, rvs_notel[:] =
        DPCA(flux_bary, λ_star_template; template = flux_star_template)

    # telluric model with updated stellar template
    est_tellurics!(
        telluric_obs,
        repeat(flux_star_template, 1, n_obs),
        var_bary,
    )
    flux_telluric_guess[:] = make_template(telluric_obs)
    _, M_tel[:, :], s_tel[:, :], _ =
        fit_gen_pca(telluric_obs; num_components=2, mu=flux_telluric_guess)
end

linear_model(θ) = (θ[1] * θ[2]) .+ θ[3]  # M * s + template
function model_prior(θ, coeffs::Vector{<:Real})
    template_mod = θ[3] .- 1
    return (coeffs[1] * sum(template_mod[template_mod.>0])) +
    (coeffs[2] * L1(template_mod)) +
    (coeffs[3] * L2(template_mod)) +
    (coeffs[4] * L1(θ[1])) +
    (coeffs[5] * L2(θ[1])) +
    L1(θ[2])
end

star_model_result = ones(size(telluric_obs))
star_model_bary = ones(size(flux_bary))
gpx_template = SOAP_gp(log_λ_star_template, SOAP_gp_ridge)
function star_model!(star_model_bary, star_model_result, M_rv, s_rv, θ)
    # α = Σ_bary \ ((M_rv * s_rv) + linear_model(θ))
    # for i in 1:n_obs
    #     # Σ_obs_bary[:,:] = cov(SOAP_gp_slow(Spectra[i].log_λ_bary), SOAP_gp_slow_bary_temp)  # 4s
    #     Σ_obs_bary[:,:] = load("E:/telfitting/Sigma_obs_bary_$i.jld2", "Σ_obs_bary")  # 1s
    #     star_model_result[:, i] = Σ_obs_bary * α[:, i]
    # end
    star_model_bary[:, :] = ((M_rv * s_rv) + linear_model(θ)) .- 1
    for i = 1:n_obs
        star_model_result[:, i] = get_mean_GP(
            gpx_template,
            view(star_model_bary, :, i),
            Spectra[i].log_λ_bary)
    end
    star_model_result .+= 1
end
star_prior(θ) = model_prior(θ, [2e-2, 1e-2, 1e2, 1e5, 1e6])

tel_model_result = ones(size(telluric_obs))
function tel_model!(tel_model_result, θ)
    tel_model_result[:, :] = linear_model(θ)
end
tel_prior(θ) = model_prior(θ, [2e2, 1e2, 1e3, 1e3, 1e6])

rv_model_bary = ones(size(star_model_bary))
rv_model_result = ones(size(telluric_obs))
function rv_model!(rv_model_bary, rv_model_result, M_rv, flux_star_template, θ)
    M_rv[:] = calc_doppler_component_RVSKL(λ_star_template, flux_star_template)
    rv_model_bary[:, :] = M_rv * θ
    for i = 1:n_obs
        rv_model_result[:, i] = get_mean_GP(
            gpx_template,
            view(rv_model_bary, :, i),
            Spectra[i].log_λ_bary)
    end
end

function loss(telluric_obs, obs_var, θ)
    tel_model!(tel_model_result, view(θ, 1:3))
    star_model!(star_model_bary, star_model_result, M_rv, s_rv, view(θ, 4:6))
    return sum((((tel_model_result .* star_model_result) - flux_obs) .^ 2) ./ obs_var) + tel_prior(view(θ, 1:3)) + star_prior(view(θ, 4:6))
end
function tel_loss(telluric_obs, obs_var, θ)
    tel_model!(tel_model_result, θ)
    return sum((((tel_model_result .* star_model_result) - flux_obs) .^ 2) ./ obs_var) + tel_prior(view(θ, 1:3)) + star_prior(view(θ, 4:6))
end
function loss(telluric_obs, obs_var, θ)
    tel_model!(tel_model_result, view(θ, 1:3))
    star_model!(star_model_bary, star_model_result, M_rv, s_rv, view(θ, 4:6))
    return sum((((tel_model_result .* star_model_result) - flux_obs) .^ 2) ./ obs_var) + tel_prior(view(θ, 1:3)) + star_prior(view(θ, 4:6))
end

function θ_holder!(θ_holder, θ, inds)
    for i in 1:length(θ_holder)
        θ_holder[i][:,:] = reshape(θ[inds[i]], size(θ_holder[i]))
    end
end
function θ_holder_to_θ(θ_holder, inds)
    θ = zeros(sum([length(i) for i in θ_holder]))
    for i in 1:length(θ_holder)
        θ[inds[i]] = collect(Iterators.flatten(θ_holder[i][:,:]))
    end
    return θ
end

function f(θ)
    θ_holder!(θ_holder, θ, inds_hold)
    return loss(telluric_obs, obs_var, θ)
end
function g!(G, θ)
    θ_holder!(θ_holder, θ, inds_hold)
    grads = gradient((θ_holder) -> loss(telluric_obs, obs_var, θ_holder), θ_holder)[1]
    for i in 1:length(inds_hold)
        G[inds_hold[i]] = collect(Iterators.flatten(grads[i]))
    end
end

_, M_star, s_star, _, rvs_notel = DPCA(flux_bary, λ_star_template; template = flux_star_template)


@time reset_fit!()

M_rv, s_rv = M_star[:, 1], s_star[1, :]'
M_star_var, s_star_var = M_star[:, 2:3], s_star[2:3, :]
# s_rv ./= 5
# s_star_var ./= 5

θ_holder = [M_tel, s_tel, flux_telluric_guess, M_star_var, s_star_var, flux_star_template]

inds_hold = [1:length(θ_holder[1])]
for i in 2:length(θ_holder)
    append!(inds_hold, [(inds_hold[i-1][end]+1):(inds_hold[i-1][end]+length(θ_holder[i]))])
end

resid_stds = [std((rvs_notel - rvs_kep_nu) - rvs_activ_no_noise)]
losses = [loss(telluric_obs, obs_var, θ_holder)]
tracker = 0

status_plot()

println("guess $tracker, std=$(round(std(rvs_notel - rvs_kep_nu - rvs_activ_no_noise), digits=5))")
for i = 1:6
    tracker += 1
    println("guess $tracker")

    # fit tellurics
    @time optimize(f_tel, g_tel!, θ_holder_to_θ(θ_holder_tel, inds_tel), LBFGS(),
        Optim.Options(iterations=30, f_tol=1e-3, g_tol=1e5))
    M_tel[:,:], s_tel[:,:], flux_telluric_guess[:] = θ_holder_tel
    est_flux_bary!(flux_bary, var_bary, tel_model(θ_holder_tel))

    # fit RV
    _, M_rv[:], s_rv[:], _, rvs_notel[:] = DPCA(flux_bary, λ_star_template;
        template=flux_star_template, num_components=1)

    # fit stellar variability
    @time optimize(f_star, g_star!, θ_holder_to_θ(θ_holder_star, inds_star), LBFGS(),
        Optim.Options(iterations=30, f_tol=1e-3, g_tol=1e5))
    M_star_var[:,:], s_star_var[:,:], flux_star_template[:] = θ_holder_star
    est_tellurics!(telluric_obs, star_model(M_rv, s_rv, θ_holder_star), var_bary)

    append!(resid_stds, [std(rvs_notel - rvs_kep_nu - rvs_activ_no_noise)])
    append!(tel_losses, [tel_loss(telluric_obs, obs_var, θ_holder_tel)])
    append!(star_losses, [star_loss(M_rv, s_rv, flux_bary, var_bary, θ_holder_star)])

    println("tel loss, data  = $(tel_loss_data(telluric_obs, obs_var, θ_holder_tel)), prior = $(tel_prior(θ_holder_tel))")
    println("star loss, data = $(star_loss_data(M_rv, s_rv, flux_bary, var_bary, θ_holder_star)), prior = $(star_prior(θ_holder_star))")
    println("rv std          = $(round(std((rvs_notel - rvs_kep_nu) - rvs_activ_no_noise), digits=5))")
    status_plot()
end
plot(resid_stds; xlabel="iter", ylabel="predicted RV - active RV RMS", legend=false)
plot(tel_losses; xlabel="iter", ylabel="tel loss", legend=false)
plot(star_losses; xlabel="iter", ylabel="star loss", legend=false)

## Extra plots

# Compare RV differences to actual RVs from activity
predict_plot = plot_rv()
plot!(predict_plot, times_nu, rvs_activ_noisy .- mean(rvs_activ_noisy), st=:scatter, ms=3, color=:red, label="Actvity (with obs. SNR and resolution)")
plot!(predict_plot, times_nu, rvs_notel - rvs_kep_nu .- mean(rvs_notel - rvs_kep_nu), st=:scatter, ms=3, color=:green, label="Third Guess - Truth")
png(predict_plot, "figs/test9.png")

# Compare second guess at RVs to true signal
predict_plot = plot_rv()
plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Truth")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="First Guess")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel, st=:scatter, ms=3, color=:green, label="Third Guess")
png(predict_plot, "figs/test10.png")
