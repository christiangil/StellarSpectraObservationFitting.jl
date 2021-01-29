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
build_gp_slow(θ) = to_sde(build_gp_base(θ))

plot_spectrum(; kwargs...) = plot(; xlabel = "Wavelength (nm)", ylabel = "Continuum Normalized Flux", dpi = 400, kwargs...)
plot_rv(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "RV (m/s)", dpi = 400, kwargs...)

function get_marginal_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::AbstractVector,
    xs::AbstractVector;
    og_gp=false)
    if typeof(og_gp)==Bool
        gp_post = posterior(finite_GP, ys)
    else
        gp_post = og_gp | (finite_GP ← ys)
    end
    gpx_post = gp_post(xs)
    return TemporalGPs.marginals(gpx_post)
end


function get_mean_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::AbstractVector,
    xs::AbstractVector; kwargs...)
    return mean.(get_marginal_GP(finite_GP, ys, xs; kwargs...))
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
star_template_res = 2 * sqrt(2) * obs_resolution

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
for i in 1:n_obs # 13s
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
    for i in 1:size(fluxes, 2) # 15s
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
png(predict_plot, "figs/spectrum.png")

# Compare first guess at RVs to true signal
plot_times = linspace(times_nu[1], times_nu[end], 1000)
plot_rvs_kep = ustrip.(planet_ks.(plot_times .* 1u"d"))
predict_plot = plot_rv()
plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
png(predict_plot, "figs/model_0_phase.png")

# Compare RV differences to actual RVs from activity
rvs_kep_nu = ustrip.(rvs_kep)
predict_plot = plot_rv()
plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
plot!(predict_plot, times_nu, rvs_naive - rvs_kep_nu, st=:scatter, ms=3, color=:blue, label="Before model")
png(predict_plot, "figs/model_0.png")

## 2) Use estimated stellar template to estimate tellurics

function est_tellurics!(
    tellurics::Matrix{T},
    fluxes::Matrix{T},
    vars::Matrix{T},
) where {T<:Real}
    for i in 1:n_obs # 13s
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
plot!(predict_plot, obs_λ, true_tels_mean / norm(true_tels_mean), st=:line, lw=1.5, label="Average telluric")
plot!(predict_plot, obs_λ, μ_tel / norm(μ_tel), st=:line, lw=1, label="Intial telluric model mean", alpha=0.8)
# plot!(predict_plot, obs_λ, M_tel[:,1], st=:line, lw=1, label="Telluric guess component 1", alpha=0.5)
# plot!(predict_plot, obs_λ, M_tel[:,2], st=:line, lw=1, label="Telluric guess component 2", alpha=0.5)
png(predict_plot, "figs/mean_tel.png")

## 3) Use estimated tellurics template to make first pass at improving RVs

est_flux_bary!(flux_bary, var_bary, repeat(μ_tel, 1, n_obs))
# est_flux_bary!(flux_bary, var_bary, telluric_obs .+ M_tel * s_tel)

μ_star = make_template(flux_bary)
_, M_star, s_star, _, rvs_notel = DPCA(flux_bary, λ_star_template; template = μ_star)

# Compare second guess at RVs to true signal
predict_plot = plot_rv()
plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
png(predict_plot, "figs/model_1_phase.png")

# Compare RV differences to actual RVs from activity
predict_plot = plot_rv()
plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
plot!(predict_plot, times_nu, rvs_notel - rvs_kep_nu, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
png(predict_plot, "figs/model_1.png")

std((rvs_naive - rvs_kep_nu) - rvs_activ_no_noise)
std((rvs_notel - rvs_kep_nu) - rvs_activ_no_noise)
std((rvs_naive - rvs_kep_nu) - rvs_activ_noisy)
std((rvs_notel - rvs_kep_nu) - rvs_activ_noisy)
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
for i in 1:size(obs_var, 2) # 15s
    obs_var[:, i] = Spectra[i].var_obs
end

L1(thing) = sum(abs.(thing))
L2(thing) = sum(thing .* thing)

function status_plot(θ_tot; plot_epoch=1, kwargs...)
    tel_model_result = tel_model(view(θ_tot, 1:3))
    star_model_result = star_model(view(θ_tot, 4:6); epochs=plot_epoch, kwargs...)
    rv_model_result = _rv_model(M_rv, view(θ_tot, 7); epochs=plot_epoch, kwargs...)

    l = @layout [a; b]
    # predict_plot = plot_spectrum(; legend = :bottomleft, size=(800,1200), layout = l)
    # predict_plot = plot_spectrum(; xlim=(627.8,628.3), legend=:bottomleft, size=(800,1200), layout = l) # o2
    # predict_plot = plot_spectrum(; xlim=(651.5,652), legend=:bottomleft, size=(800,1200), layout = l)  # h2o
    predict_plot = plot_spectrum(; xlim = (647, 656), legend = :bottomleft, size=(800,1200), layout = l)  # h2o
    plot!(predict_plot[1], obs_λ, true_tels[:, plot_epoch], label="true tel")
    if !using_GPs
        plot!(predict_plot[1], obs_λ, flux_obs[:, plot_epoch] ./ (star_model_result[:, plot_epoch] + rv_model_result[:, plot_epoch]), label="predicted tel", alpha = 0.5)
    else
        plot!(predict_plot[1], obs_λ, flux_obs[:, plot_epoch] ./ (star_model_result[1] + rv_model_result[1]), label="predicted tel", alpha = 0.5)
    end
    plot!(predict_plot[1], obs_λ, tel_model_result[:, plot_epoch], label="model tel: $tracker", alpha = 0.5)
    plot_bary_λs = exp.(Spectra[plot_epoch].log_λ_bary)
    plot!(predict_plot[2], plot_bary_λs, flux_obs[:, plot_epoch] ./ true_tels[:, plot_epoch], label="true star", )
    plot!(predict_plot[2], plot_bary_λs, flux_obs[:, plot_epoch] ./ tel_model_result[:, plot_epoch], label="predicted star", alpha = 0.5)
    if !using_GPs
        plot!(predict_plot[2], plot_bary_λs, star_model_result[:, plot_epoch] + rv_model_result[:, plot_epoch], label="model star: $tracker", alpha = 0.5)
    else
        plot!(predict_plot[2], plot_bary_λs, star_model_result[1] + rv_model_result[1], label="model star: $tracker", alpha = 0.5)
    end
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
prior_factor = 10
tel_prior(θ) = model_prior(θ, prior_factor .* [2e2, 1e2, 1e3, 1e3, 1e6])

lower_inds = zeros(Int, size(telluric_obs))
ratios = zeros(size(telluric_obs))
for j in 1:size(telluric_obs, 2)
    lower_inds[:, j] = searchsortednearest(log_λ_star_template, Spectra[j].log_λ_bary; lower=true)
    for i in 1:size(telluric_obs, 1)
        x0 = log_λ_star_template[lower_inds[i, j]]
        x1 = log_λ_star_template[lower_inds[i, j]+1]
        x = Spectra[j].log_λ_bary[i]
        ratios[i, j] = (x - x0) / (x1 - x0)
    end
    lower_inds[:, j] .+= (j - 1) * len_bary
end

# SOAP_gp_slow = build_gp_slow(θ1)
# SOAP_gp_slow_bary_temp = SOAP_gp_slow(log_λ_star_template)

gpx_template = build_gp(θ1)(log_λ_star_template, SOAP_gp_ridge)
# gpx_template = build_gp_slow(θ1)(log_λ_star_template, SOAP_gp_ridge)
# SOAP_gp_stheno = build_gp_base(θ1)
# gpx_template = SOAP_gp_stheno(log_λ_star_template, 1e10*SOAP_gp_ridge)
using_GPs = false
if !using_GPs
    function spectra_interp(bary_vals; epochs=1:n_obs, method="linear")
        @assert method in ["linear", "alt linear", "GP", "sinc"]
        if method == "linear"
            model_result = (bary_vals[lower_inds] .* (1 .- ratios)) + (bary_vals[lower_inds .+ 1] .* (ratios))
        else
            model_result = zeros(size(telluric_obs))
            if method == "alt linear"
                for i in epochs
                    model_result[:, i] = LinearInterpolation(log_λ_star_template,
                        view(bary_vals, :, i)).(Spectra[i].log_λ_bary)
                end
            elseif method == "GP"
                for i in epochs
                    model_result[:, i] = get_mean_GP(
                        gpx_template,
                        view(bary_vals, :, i) .- 1,
                        Spectra[i].log_λ_bary) .+ 1
                end
            elseif method == "sinc"
                for i in epochs
                    model_result[:, i] = spectra_interpolate(log_λ_star_template,
                        Spectra[i].log_λ_bary,
                        view(bary_vals, :, i))
                end
            end
        end
        return model_result
    end
else
    function spectra_interp(bary_vals; epochs=1:n_obs, method="GP")
        @assert method in ["linear", "alt linear", "GP", "sinc"]
        if method == "linear"
            model_result = [(bary_vals[lower_inds[:, i]] .* (1 .- ratios[:, i])) + (bary_vals[lower_inds[:, i] .+ 1] .* (ratios[:, i])) for i in epochs]
        elseif method == "alt linear"
            model_result = [LinearInterpolation(log_λ_star_template,
                view(bary_vals, :, i)).(Spectra[i].log_λ_bary) for i in epochs]
        elseif method == "GP"
            model_result = [get_mean_GP(
                gpx_template,
                view(bary_vals, :, i) .- 1,
                Spectra[i].log_λ_bary) .+ 1 for i in epochs]
        elseif method == "sinc"
            model_result = [spectra_interpolate(log_λ_star_template,
                Spectra[i].log_λ_bary,
                view(bary_vals, :, i)) for i in epochs]
        end
        return model_result
    end
end

star_model_result = ones(size(telluric_obs))
star_model(θ; kwargs...) = spectra_interp(linear_model(θ); kwargs...)
# star_prior(θ) = model_prior(θ, [2e-2, 1e-2, 1e2, 1e5, 1e6])
star_prior(θ) = model_prior(θ, prior_factor .* [2e-2, 1e-2, 1e2, 1e5, 1e6])

rv_model_result = ones(size(telluric_obs))
rv_model(μ_star, θ; kwargs...) = _rv_model(calc_doppler_component_RVSKL(λ_star_template, μ_star), θ; kwargs...)
_rv_model(M_rv, θ; kwargs...) = spectra_interp(M_rv * θ[1]; kwargs...)
M_rv, s_rv = M_star[:, 1], s_star[1, :]'

if !using_GPs
    _loss(tel_model_result, star_model_result, rv_model_result) =
        sum((((tel_model_result .* (star_model_result + rv_model_result)) - flux_obs) .^ 2) ./ obs_var)
else
    function _loss(tel_model_result, star_model_result, rv_model_result)
        ans = 0
        for i in length(star_model_result)
            ans += sum((((tel_model_result[:, i] .* (star_model_result[i] + rv_model_result[i])) - flux_obs[:, i]) .^ 2) ./ obs_var)
        end
        return ans
    end
end

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

function θ_holder!(θ_holder, θ, inds)
    for i in 1:length(inds)
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

function f(θ, θ_holder, inds, loss_func; kwargs...)
    θ_holder!(θ_holder, θ, inds; kwargs...)
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

# status_plot(θ_tot; plot_epoch=plot_epoch, method="GP")
# status_plot(θ_tot; plot_epoch=plot_epoch, method="linear")
# status_plot(θ_tot; plot_epoch=plot_epoch, method="alt linear")
status_plot(θ_tot; plot_epoch=plot_epoch)

OOptions = Optim.Options(iterations=10, f_tol=1e-3, g_tol=1e5)

tel_model_result[:, :] = tel_model(θ_tel)
rv_model_result = rv_model(μ_star, θ_rv)
star_model_result = star_model(θ_star)

# optimize(f_tel, g_tel!, θ_holder_to_θ(θ_tel, inds_tel), LBFGS(), OOptions)
# optimize(f_rv, g_rv!, θ_holder_to_θ(θ_rv, inds_rv), LBFGS(), OOptions)
# optimize(f_star, g_star!, θ_holder_to_θ(θ_star, inds_star), LBFGS(), OOptions)

println("guess $tracker, std=$(round(std(rvs_notel - rvs_kep_nu - rvs_activ_no_noise), digits=5))")
rvs_notel_opt = copy(rvs_notel)
@time for i in 1:3
    tracker += 1
    println("guess $tracker")

    optimize(f_star, g_star!, θ_holder_to_θ(θ_star, inds_star), LBFGS(), OOptions)
    star_model_result[:, :] = star_model(θ_star)
    M_rv[:, :] = calc_doppler_component_RVSKL(λ_star_template, μ_star)

    optimize(f_tel, g_tel!, θ_holder_to_θ(θ_tel, inds_tel), LBFGS(), OOptions)
    tel_model_result[:, :] = tel_model(θ_tel)

    optimize(f_rv, g_rv!, θ_holder_to_θ(θ_rv, inds_rv), LBFGS(), OOptions)
    rv_model_result[:, :] = rv_model(μ_star, θ_rv)
    rvs_notel_opt[:] = (s_rv .* light_speed_nu)'

    append!(resid_stds, [std(rvs_notel_opt - rvs_kep_nu - rvs_activ_no_noise)])
    append!(losses, [loss(θ_tot)])

    println("loss   = $(loss(θ_tot))")
    println("rv std = $(round(std((rvs_notel_opt - rvs_kep_nu) - rvs_activ_no_noise), digits=5))")
    status_plot(θ_tot; plot_epoch=plot_epoch)
end

plot(resid_stds; xlabel="iter", ylabel="predicted RV - active RV RMS", legend=false)
plot(losses; xlabel="iter", ylabel="loss", legend=false)

## Extra plots

# Compare RV differences to actual RVs from activity
predict_plot = plot_rv()
plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
plot!(predict_plot, times_nu, rvs_notel - rvs_kep_nu, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
plot!(predict_plot, times_nu, rvs_notel_opt - rvs_kep_nu, st=:scatter, ms=3, color=:darkgreen, label="After optimization")
png(predict_plot, "figs/model_2.png")

# Compare second guess at RVs to true signal
predict_plot = plot_rv()
plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel_opt, st=:scatter, ms=3, color=:darkgreen, label="After optimization")
png(predict_plot, "figs/model_2_phase.png")

# predict_plot = plot_spectrum(; xlim=(627.8,628.3)) # o2
# predict_plot = plot_spectrum(; xlim = (647, 656))  # h2o
# predict_plot = plot_spectrum(; xlim=(651.5,652))  # h2o
predict_plot = plot_spectrum(; title="Stellar model")
plot!(λ_star_template, μ_star; label="μ")
plot!(λ_star_template, M_star_var[:,1]; label="basis 1")
plot!(λ_star_template, M_star_var[:,2]; label="basis 2")
png(predict_plot, "figs/model_star_basis.png")

plot_scores(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "Weights", dpi = 400, kwargs...)
predict_plot = plot_scores(; title="Stellar model")
scatter!(times_nu, s_star_var[1, :]; label="weights 1")
scatter!(times_nu, s_star_var[2, :]; label="weights 2")
png(predict_plot, "figs/model_star_weights.png")

scatter(times_nu, s_tel')


scatter(times_nu, (rvs_notel_opt - rvs_kep_nu) ./ (std(rvs_notel_opt - rvs_kep_nu)/std(s_star_var[1,:])))

predict_plot = plot_spectrum(; title="Telluric model")
plot!(obs_λ, μ_tel; label="μ")
plot!(obs_λ, M_tel[:,1]; label="basis 1")
plot!(obs_λ, M_tel[:,2]; label="basis 2")
png(predict_plot, "figs/model_tel_basis.png")

predict_plot = plot_scores(; title="Telluric model")
scatter!(times_nu, s_tel[1, :]; label="weights 1")
scatter!(times_nu, s_tel[2, :]; label="weights 2")
scatter!(times_nu, airmasses .- 3; label="airmasses")
png(predict_plot, "figs/model_tel_weights.png")
