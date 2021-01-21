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
include("../src/interpolation_functions.jl")
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
n_star_template = Int(round((max_temp_wav - min_temp_wav) * star_template_res / sqrt(max_temp_wav * min_temp_wav)))
log_λ_star_template = RegularSpacing(log(min_temp_wav), (log(max_temp_wav) - log(min_temp_wav)) / n_star_template, n_star_template)
λ_star_template =  exp.(log_λ_star_template)

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
μ_star = make_template(flux_bary)
_, _, _, _, rvs_naive = DPCA(flux_bary, λ_star_template; template=μ_star, num_components=1)

predict_plot = plot_spectrum(; legend = :bottomright, size=(1200, 800))
# predict_plot = plot_spectrum(; xlim = (627.8, 628.3), legend = :bottomright)
plot!(predict_plot, λ_star_template, μ_star, st=:line, lw=1, label="Predicted Quiet")
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

est_tellurics!(telluric_obs, repeat(μ_star, 1, n_obs), ones(size(flux_bary)) * SOAP_gp_ridge)
μ_tel = make_template(telluric_obs)
_, M_tel, s_tel, _ = fit_gen_pca(telluric_obs; num_components=2, mu=μ_tel)

## telluric sanity check plots

# Compare telluric guess to actual tellurics
predict_plot = plot_spectrum(; legend = :bottomright, size=(1200, 800))
# zoom in on o2 lines
# predict_plot = plot_spectrum(; xlim=(627.8,628.3), legend=:bottomright)
# zoom in on h2o lines
# predict_plot = plot_spectrum(; xlim=(651.5,652), legend=:bottomright)
plot!(predict_plot, obs_λ, true_tels_mean / norm(true_tels_mean), st=:line, lw=1.5, label="Average Telluric")
plot!(predict_plot, obs_λ, μ_tel / norm(μ_tel), st=:line, lw=1, label="Telluric guess mean", alpha=0.8)
# plot!(predict_plot, obs_λ, M_tel[:,1], st=:line, lw=1, label="Telluric guess component 1", alpha=0.5)
# plot!(predict_plot, obs_λ, M_tel[:,2], st=:line, lw=1, label="Telluric guess component 2", alpha=0.5)
png(predict_plot, "figs/test6.png")

## 3) Use estimated tellurics template to make first pass at improving RVs

est_flux_bary!(flux_bary, var_bary, repeat(μ_tel, 1, n_obs))
# est_flux_bary!(flux_bary, var_bary, telluric_obs .+ M_tel * s_tel)

μ_star = make_template(flux_bary)
_, M_star, s_star, _, rvs_notel = DPCA(flux_bary, λ_star_template; template = μ_star)

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

est_tellurics!(telluric_obs, repeat(μ_star, 1, n_obs), var_bary)
μ_tel = make_template(telluric_obs)
_, M_tel, s_tel, _ = fit_gen_pca(telluric_obs; num_components=2, mu=μ_tel)

plot_epoch = 60
predict_plot = plot_spectrum(; legend = :bottomright, size=(1200, 800))
# zoom in on o2 lines
# predict_plot = plot_spectrum(; xlim=(627.8,628.3), legend=:bottomright)
# zoom in on h2o lines
# predict_plot = plot_spectrum(; xlim=(651.5,652), legend=:bottomright, size=(1200,800))
plot!(predict_plot, obs_λ, true_tels[:, plot_epoch], #- μ_tel,
    st=:line, lw=2, label="actual tels - telluric template")
plot!(predict_plot, obs_λ, telluric_obs[:, plot_epoch], #- μ_tel,
    st=:line, lw=1.5, label="obs / stellar - telluric template", alpha=0.8)
plot!(predict_plot, obs_λ, (M_tel*s_tel)[:, plot_epoch] + μ_tel, st=:line, lw=1, label="basis * scores", alpha=0.8)


## fit star

obs_var = zeros(size(telluric_obs))
for i = 1:size(obs_var, 2) # 15s
    obs_var[:, i] = Spectra[i].var_obs
end

L1(thing) = sum(abs.(thing))
L2(thing) = sum(thing .* thing)

function status_plot(θ_tot; plot_epoch=1)
    tel_model_result[:, :] = tel_model(view(θ_tot, 1:3))
    star_model_result[:, :] = star_model(view(θ_tot, 4:6))
    rv_model_result[:, :] = _rv_model(M_rv, view(θ_tot, 7))

    l = @layout [a; b]
    # predict_plot = plot_spectrum(; legend = :bottomleft, size=(800,1200), layout = l)
    # predict_plot = plot_spectrum(; xlim=(627.8,628.3), legend=:bottomleft, size=(800,1200), layout = l) # o2
    # predict_plot = plot_spectrum(; xlim=(651.5,652), legend=:bottomleft, size=(800,1200), layout = l)  # h2o
    predict_plot = plot_spectrum(; xlim = (647, 656), legend = :bottomleft, size=(800,1200), layout = l)  # h2o
    plot!(predict_plot[1], obs_λ, true_tels[:, plot_epoch], label="true tel")
    plot!(predict_plot[1], obs_λ, flux_obs[:, plot_epoch] ./ (star_model_result[:, plot_epoch] + rv_model_result[:, plot_epoch]), label="predicted tel", alpha = 0.5)
    plot!(predict_plot[1], obs_λ, tel_model_result[:, plot_epoch], label="model tel: $tracker", alpha = 0.5)
    plot_bary_λs = exp.(Spectra[plot_epoch].log_λ_bary)
    plot!(predict_plot[2], plot_bary_λs, flux_obs[:, plot_epoch] ./ true_tels[:, plot_epoch], label="true star", )
    plot!(predict_plot[2], plot_bary_λs, flux_obs[:, plot_epoch] ./ tel_model_result[:, plot_epoch], label="predicted star", alpha = 0.5)
    plot!(predict_plot[2], plot_bary_λs, star_model_result[:, plot_epoch] + rv_model_result[:, plot_epoch], label="model star: $tracker", alpha = 0.5)
    display(predict_plot)
end

function reset_fit!()
    telluric_obs[:, :] .= 1  # part of model
    flux_bary[:, :] .= 1  # interpolation / telluric_obs
    var_bary[:, :] .= 1

    # model everything as stellar variability
    est_flux_bary!(flux_bary, var_bary, telluric_obs)
    μ_star[:] = make_template(flux_bary)
    _, M_star[:, :], s_star[:, :], _, rvs_notel[:] =
        DPCA(flux_bary, λ_star_template; template=μ_star)

    # telluric model with stellar template
    est_tellurics!(telluric_obs, repeat(μ_star, 1, n_obs), var_bary)
    μ_tel[:] = make_template(telluric_obs)
    _, M_tel[:, :], s_tel[:, :], _ =
        fit_gen_pca(telluric_obs; num_components=2, mu=μ_tel)

    # stellar model with telluric template
    est_flux_bary!(flux_bary, var_bary, repeat(μ_tel, 1, n_obs))
    μ_star[:] = make_template(flux_bary)
    _, M_star[:, :], s_star[:, :], _, rvs_notel[:] =
        DPCA(flux_bary, λ_star_template; template = μ_star)

    # telluric model with updated stellar template
    est_tellurics!(telluric_obs, repeat(μ_star, 1, n_obs), var_bary)
    μ_tel[:] = make_template(telluric_obs)
    _, M_tel[:, :], s_tel[:, :], _ =
        fit_gen_pca(telluric_obs; num_components=2, mu=μ_tel)
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

tel_model_result = ones(size(telluric_obs))
tel_model(θ) = linear_model(θ)
# tel_prior(θ) = model_prior(θ, [2e2, 1e2, 1e3, 1e3, 1e6])
tel_prior(θ) = model_prior(θ, 0. * [2e2, 1e2, 1e3, 1e3, 1e6])

lower_inds = zeros(Int, size(telluric_obs))
ratios = zeros(size(telluric_obs))
for j in 1:size(telluric_obs, 2)
    lower_inds[:, j] = searchsortednearest(log_λ_star_template, Spectra[j].log_λ_bary; lower=true)
    for i in 1:size(telluric_obs, 1)
        x1 = log_λ_star_template[lower_inds[i, j]]
        x2 = log_λ_star_template[lower_inds[i, j]+1]
        x3 = Spectra[j].log_λ_bary[i]
        ratios[i, j] = (x3 - x1) / (x2 - x1)
    end
    lower_inds[:, j] += (j - 1) * len_bary
end

# gpx_template = SOAP_gp(log_λ_star_template, SOAP_gp_ridge)
function spectra_interp(bary_vals)
    model_result = (bary_vals[lower_inds] .* ratios) + (bary_vals[lower_inds .+ 1] .* (1 .- ratios))
    # model_result = zeros(size(telluric_obs))
    # if method == "alt linear"
    #     for i = 1:n_obs
    #         model_result[:, i] = LinearInterpolation(log_λ_star_template,
    #             view(bary_vals, :, i); extrapolation_bc=Line()).(Spectra[i].log_λ_bary)
    #     end
    # elseif method == "GP"
    #     for i = 1:n_obs
    #         model_result[:, i] = get_mean_GP(
    #             gpx_template,
    #             view(bary_vals, :, i) .- 1,
    #             Spectra[i].log_λ_bary) .+ 1
    #     end
    # elseif method == "sinc"
    #     for i = 1:n_obs
    #         model_result[:, i] = spectra_interpolate(log_λ_star_template,
    #             Spectra[i].log_λ_bary,
    #             view(bary_vals, :, i))
    #     end
    # end
    return model_result
end

star_model_result = ones(size(telluric_obs))
star_model(θ; kwargs...) = spectra_interp(linear_model(θ); kwargs...)
# star_prior(θ) = model_prior(θ, [2e-2, 1e-2, 1e2, 1e5, 1e6])
star_prior(θ) = model_prior(θ, 0 .* [2e-2, 1e-2, 1e2, 1e5, 1e6])

rv_model_result = ones(size(telluric_obs))
rv_model(μ_star, θ; kwargs...) = _rv_model(calc_doppler_component_RVSKL(λ_star_template, μ_star), θ; kwargs...)
_rv_model(M_rv, θ; kwargs...) = spectra_interp(M_rv * θ[1]; kwargs...)
M_rv, s_rv = M_star[:, 1], s_star[1, :]'
rv_model_result[:, :] = rv_model(μ_star, [s_rv])

_loss(tel_model_result, star_model_result, rv_model_result) =
    sum((((tel_model_result .* (star_model_result + rv_model_result)) - flux_obs) .^ 2) ./ obs_var)

function loss(θ;
    tel_model_result = tel_model(view(θ, 1:3)),
    star_model_result = star_model(view(θ, 4:6)),
    rv_model_result = rv_model(μ_star, view(θ, 7)))
    return _loss(tel_model_result, star_model_result, rv_model_result)
end
loss_tel(θ) = _loss(tel_model(θ), star_model_result, rv_model_result) +
    tel_prior(θ)
loss_star(θ) = _loss(tel_model_result, star_model(θ), rv_model_result) +
    star_prior(θ)
loss_rv(θ) = _loss(tel_model_result, star_model_result, _rv_model(M_rv, θ))

function θ_holder!(θ_holder, θ, inds; recalc_M_rv::Bool=true)
    for i in 1:length(inds)
        θ_holder[i][:,:] = reshape(θ[inds[i]], size(θ_holder[i]))
    end
    if recalc_M_rv
        M_rv[:, :] = calc_doppler_component_RVSKL(λ_star_template, μ_star)
    end
end
function θ_holder_to_θ(θ_holder, inds)
    θ = zeros(sum([length(i) for i in θ_holder]))
    for i in 1:length(θ_holder)
        θ[inds[i]] = collect(Iterators.flatten(θ_holder[i][:,:]))
    end
    return θ
end

@time reset_fit!()

M_rv, s_rv = M_star[:, 1], s_star[1, :]'
M_star_var, s_star_var = M_star[:, 2:3], s_star[2:3, :]

s_star_var ./= 5

θ_tot = [M_tel, s_tel, μ_tel, M_star_var, s_star_var, μ_star, s_rv]
θ_tel = [M_tel, s_tel, μ_tel]
θ_star = [M_star_var, s_star_var, μ_star]
θ_rv = [s_rv]

function relevant_inds(θ_hold)
    inds = [1:length(θ_hold[1])]
    for i in 2:length(θ_hold)
        append!(inds, [(inds[i-1][end]+1):(inds[i-1][end]+length(θ_hold[i]))])
    end
    return inds
end
inds_tel = relevant_inds(θ_tel)
inds_star = relevant_inds(θ_star)
inds_rv = relevant_inds(θ_rv)

function f(θ, θ_holder, inds, loss_func)
    θ_holder!(θ_holder, θ, inds)
    return loss_func(θ_holder)
end
f_tel(θ) = f(θ, θ_tel, inds_tel, loss_tel)
f_star(θ) = f(θ, θ_star, inds_star, loss_star)
f_rv(θ) = f(θ, θ_rv, inds_rv, loss_rv)

function g!(G, θ, θ_holder, inds, loss_func)
    θ_holder!(θ_holder, θ, inds)
    grads = gradient((θ_hold) -> loss_func(θ_hold), θ_holder)[1]
    for i in 1:length(inds)
        G[inds[i]] = collect(Iterators.flatten(grads[i]))
    end
end
g_tel!(G, θ) = g!(G, θ, θ_tel, inds_tel, loss_tel)
g_star!(G, θ) = g!(G, θ, θ_star, inds_star, loss_star)
g_rv!(G, θ) = g!(G, θ, θ_rv, inds_rv, loss_rv)

resid_stds = [std((rvs_notel - rvs_kep_nu) - rvs_activ_no_noise)]
losses = [loss(θ_tot)]
tracker = 0

# plot(M_star_var)
# plot(s_star_var')
# plot(μ_star)
#
# plot(M_tel)
# plot(s_tel')
# plot(μ_tel)
#
# plot(M_rv)
# plot(s_rv')
#
# plot(star_model(θ_star)[:, 13])

status_plot(θ_tot; plot_epoch=10)
OOptions = Optim.Options(iterations=10, f_tol=1e-3, g_tol=1e5)

println("guess $tracker, std=$(round(std(rvs_notel - rvs_kep_nu - rvs_activ_no_noise), digits=5))")
for i = 1:4
    tracker += 1
    println("guess $tracker")

    optimize(f_star, g_star!, θ_holder_to_θ(θ_star, inds_star), LBFGS(), OOptions)
    optimize(f_tel, g_tel!, θ_holder_to_θ(θ_tel, inds_tel), LBFGS(), OOptions)
    optimize(f_rv, g_rv!, θ_holder_to_θ(θ_rv, inds_rv), LBFGS(), OOptions)

    rvs_notel[:] = (s_rv .* light_speed_nu)'
    append!(resid_stds, [std(rvs_notel - rvs_kep_nu - rvs_activ_no_noise)])
    append!(losses, [loss(θ_tot)])

    println("loss   = $(loss(θ_tot))")
    println("rv std = $(round(std((rvs_notel - rvs_kep_nu) - rvs_activ_no_noise), digits=5))")
    status_plot(θ_tot)
end
plot(resid_stds; xlabel="iter", ylabel="predicted RV - active RV RMS", legend=false)
plot(losses; xlabel="iter", ylabel="loss", legend=false)

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
