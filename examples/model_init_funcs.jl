# using Statistics
# using Zygote  # Slow
# using TemporalGPs
# using Plots
# using Optim
# using Flux

## Getting precomputed airmasses and observation times
using Distributions


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

# amplitude, lengthscale (in log lambda)
SOAP_gp_θ = [3.3270754364467443, 9.021560480866474e-5]
SOAP_gp = build_gp(SOAP_gp_θ)

function flux_tel_gp(index::Int, star::Vector{T}, vars::Vector{T}) where {T<:Real}
    return get_marginal_GP(
        SOAP_gp(Spectra[index].log_λ_obs, (Spectra[index].var_obs ./ telluric) ./ telluric),
        (Spectra[index].flux_obs ./ telluric) .- 1,
        log_λ_star_template,
    )
end

function est_flux_tel!(tellurics::Matrix{T}, stars::Matrix{T},
    vars::Matrix{T}, log_λ_star_template::Vector{T}, Spectra) where {T<:Real}
    for i in 1:n_obs # 13s
        tellurics[:, i] = Spectra[i].flux_obs ./ (get_mean_GP(
            SOAP_gp(log_λ_star_template, vars[:, i]),
            stars[:, i] .- 1,
            Spectra[i].log_λ_bary) .+ 1)
    end
end

function est_flux_bary!(
    stars::Matrix{T},
    vars::Matrix{T},
    tellurics::Matrix{T},
) where {T<:Real}
    for i in 1:size(stars, 2) # 15s
        gp = get_marginal_GP(
            SOAP_gp(Spectra[i].log_λ_bary, (Spectra[i].var_obs ./ tellurics[:, i]) ./ tellurics[:, i]),
            (Spectra[i].flux_obs ./ tellurics[:, i]) .- 1,
            log_λ_star_template)
        stars[:, i] = mean.(gp) .+ 1
        vars[:, i] = var.(gp)
    end
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
    rvs_naive = copy(rvs_notel)

    # telluric model with stellar template
    est_flux_tel!(telluric_obs, repeat(μ_star, 1, n_obs), var_bary)
    μ_tel = make_template(telluric_obs)
    _, M_tel, s_tel, _ =
        fit_gen_pca(telluric_obs; num_components=2, mu=μ_tel)

    # stellar model with telluric template
    est_flux_bary!(flux_bary, var_bary, repeat(μ_tel, 1, n_obs))
    μ_star[:] = make_template(flux_bary)
    _, M_star[:, :], s_star[:, :], _, rvs_notel[:] =
        DPCA(flux_bary, λ_star_template; template = μ_star)

    # telluric model with updated stellar template
    est_flux_tel!(telluric_obs, repeat(μ_star, 1, n_obs), var_bary)
    μ_tel[:] = make_template(telluric_obs)
    _, M_tel[:, :], s_tel[:, :], _ =
        fit_gen_pca(telluric_obs; num_components=2, mu=μ_tel)
    return telluric_obs, flux_bary, var_bary, μ_star, M_star, s_star, rvs_notel, μ_tel, M_tel, s_tel, rvs_naive
end
