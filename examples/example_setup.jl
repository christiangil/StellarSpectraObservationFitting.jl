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

include("../../src/general_functions.jl")
include("../../src/PCA_functions.jl")

make_template(matrix::Matrix{T}) where {T<:Real} = vec(median(matrix, dims=2))
make_template_mean(matrix::Matrix{T}) where {T<:Real} =
    vec(mean(matrix, dims=2))

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

function shift_log_λ(v::Unitful.Velocity, log_λ::Vector{T}) where {T<:Real}
    return log_λ .+ (log((1.0 + v / light_speed) / (1.0 - v / light_speed)) / 2)
end

# @load "E:/telfitting/telfitting_workspace.jld2"
@load "E:/telfitting/telfitting_workspace_smol.jld2" SOAP_gp SOAP_gp_ridge Spectra airmasses bary_K inds light_speed light_speed_nu max_wav min_wav n_obs obs_resolution obs_λ planet_P_nu planet_ks quiet rvs_activ_no_noise rvs_activ_noisy rvs_kep times times_nu true_tels true_tels_mean vs wavelengths θ1 λ λ_nu

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


function flux_bary_gp(index::Int, telluric::Vector{T}) where {T<:Real}
    return get_marginal_GP(
        SOAP_gp(Spectra[index].log_λ_bary, (Spectra[index].var_obs ./ telluric) ./ telluric),
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

L1(thing) = sum(abs.(thing))
L2(thing) = sum(thing .* thing)

function status_plot(θ_tot; plot_epoch=1)
    tel_model_result = tel_model(view(θ_tot, 1:3))
    star_model_result = star_model(view(θ_tot, 4:6))
    rv_model_result = _rv_model(M_rv, view(θ_tot, 7))

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

function initialize(λ_star_template, n_obs, len_obs)
    len_bary = length(λ_star_template)

    telluric_obs = ones(len_obs, n_obs)
    flux_bary = ones(len_bary, n_obs)  # interpolation / telluric_obs
    var_bary = ones(len_bary, n_obs)  # interpolation / telluric_obs

    # model everything as stellar variability
    est_flux_bary!(flux_bary, var_bary, telluric_obs)
    μ_star = make_template(flux_bary)
    _, M_star, s_star, _, rvs_notel =
        DPCA(flux_bary, λ_star_template; template=μ_star)

    # telluric model with stellar template
    est_tellurics!(telluric_obs, repeat(μ_star, 1, n_obs), var_bary)
    μ_tel = make_template(telluric_obs)
    _, M_tel, s_tel, _ =
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
    return telluric_obs, flux_bary, var_bary, μ_star, M_star, s_star, rvs_notel, μ_tel, M_tel, s_tel
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

tel_model(θ) = linear_model(θ)
