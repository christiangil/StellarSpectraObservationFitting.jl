using LinearAlgebra
using Plots

_plt_dpi = 400
_plt_size = (1920,1080)
_thickness_scaling = 2
# _theme = :default
_theme = :juno
_my_plot(; dpi = _plt_dpi, size = _plt_size, thickness_scaling=_thickness_scaling, kwargs...) =
    plot(; dpi=dpi, size=size, thickness_scaling=thickness_scaling, kwargs...)
plot_spectrum(; xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const", kwargs...) =
    _my_plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
plot_rv(; xlabel = "Time (d)", ylabel = "RV (m/s)", kwargs...) =
    _my_plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
plot_scores(; xlabel = "Time (d)", ylabel = "Weights + Const", kwargs...) =
    _my_plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
theme(_theme)
my_scatter!(plt::Union{Plots.AbstractPlot,Plots.AbstractLayout}, x::AbstractVecOrMat, y::AbstractVecOrMat; kwargs...) = scatter!(plt, x, y; markerstrokewidth=0, kwargs...)
_theme == :default ? plt_colors = palette(_theme).colors.colors : plt_colors = PlotThemes._themes[_theme].defaults[:palette].colors.colors
function plot_model_rvs(times_nu::AbstractVector{T}, rvs_naive::AbstractVector{T}, rvs_notel::AbstractVecOrMat{T}, rvs_notel_opt::AbstractVecOrMat{T}, eo_times::AbstractVector{T}, eo_rv::AbstractVector{T}, eo_rv_σ::AbstractVector{T}; display_plt::Bool=true) where {T<:Real}
    plt = plot_rv()
    my_scatter!(plt, times_nu, rvs_naive, label="Naive, std: $(round(std(rvs_naive), digits=3))", alpha = 0.2)
    my_scatter!(plt, times_nu, rvs_notel, label="Before optimization, std: $(round(std(rvs_notel), digits=3))", alpha = 0.2)
    my_scatter!(plt, eo_time, eo_rv; yerror=eo_rv_σ, label="EXPRES RVs, std: $(round(std(eo_rv), digits=3))")
    my_scatter!(plt, times_nu, rvs_notel_opt, label="After optimization, std: $(round(std(rvs_notel_opt), digits=3))", alpha = 0.7)
    if display_plt; display(plt) end
    return plt
end
function plot_model_rvs_new(times_nu::AbstractVector{T}, model_rvs::AbstractVecOrMat{T}, model_rvs_σ::AbstractVecOrMat{T}, eo_times::AbstractVector{T}, eo_rvs::AbstractVector{T}, eo_rvs_σ::AbstractVector{T}; display_plt::Bool=true, kwargs...) where {T<:Real}
    plt = plot_rv()
    my_scatter!(plt, eo_times, eo_rvs; yerror=eo_rvs_σ, label="EXPRES RVs, std: $(round(std(eo_rv), digits=3))", kwargs...)
    my_scatter!(plt, times_nu, model_rvs; yerror=model_rvs_σ, label="Model RVs, std: $(round(std(model_rvs), digits=3))", alpha = 0.7, kwargs...)
    if display_plt; display(plt) end
    return plt
end

function plot_stellar_model_bases(tfom::StellarSpectraObservationFitting.TFOrderModel; inds::UnitRange=1:size(tfom.star.lm.M, 2), display_plt::Bool=true)
    plt = plot_spectrum(; title="Stellar Model Bases", legend=:outerright)
    plot!(tfom.tel.λ, tfom.tel.lm.μ; label="μₜₑₗ", alpha=0.3, color=:white)
    plot!(tfom.star.λ, tfom.star.lm.μ; label="μₛₜₐᵣ")
    shift = 0.2
    for i in reverse(inds)
        plot!(tfom.star.λ, (tfom.star.lm.M[:, i] ./ norm(tfom.star.lm.M[:, i])) .- shift * (i - 1); label="Basis $i", color=plt_colors[((i - inds[1] + 3) % 19) + 1])
    end
    if display_plt; display(plt) end
    return plt
end
function plot_stellar_model_scores(tfom::StellarSpectraObservationFitting.TFOrderModel; inds::UnitRange=1:size(tfom.star.lm.M, 2), display_plt::Bool=true)
    plt = plot_scores(; title="Stellar Model Weights", legend=:outerright)
    shift = ceil(10 * maximum([std(tfom.star.lm.s[inds[i], :] .* norm(tfom.star.lm.M[:, inds[i]])) for i in inds])) / 2
    for i in reverse(inds)
        my_scatter!(plt, times_nu, (tfom.star.lm.s[i, :] .* norm(tfom.star.lm.M[:, i])) .- shift * (i - 1); label="Weights $i", color=plt_colors[((i - inds[1] + 3) % 19) + 1])
        hline!([-shift * (i - 1)]; label="", color=plt_colors[((i - inds[1] + 3) % 19) + 1], lw=3, alpha=0.4)
    end
    if display_plt; display(plt) end
    return plt
end

function plot_telluric_model_bases(tfom::StellarSpectraObservationFitting.TFOrderModel; inds::UnitRange=1:size(tfom.tel.lm.M, 2), display_plt::Bool=true)
    plt = plot_spectrum(; title="Telluric Model Bases", legend=:outerright)
    plot!(tfom.star.λ, tfom.star.lm.μ; label="μₛₜₐᵣ", alpha=0.3, color=:white)
    plot!(tfom.tel.λ, tfom.tel.lm.μ; label="μₜₑₗ")
    shift = 0.2
    for i in reverse(inds)
        plot!(tfom.tel.λ, (tfom.tel.lm.M[:, i] ./ norm(tfom.tel.lm.M[:, i])) .- shift * (i - 1); label="Basis $i", color=plt_colors[((i - inds[1] + 3) % 19) + 1])
    end
    if display_plt; display(plt) end
    return plt
end
function plot_telluric_model_scores(tfom::StellarSpectraObservationFitting.TFOrderModel; inds::UnitRange=1:size(tfom.tel.lm.M, 2), display_plt::Bool=true)
    plt = plot_scores(; title="Telluric Model Weights", legend=:outerright)
    my_scatter!(plt, times_nu, airmasses; label="Airmasses")
    hline!([1]; label="", color=plt_colors[1], lw=3, alpha=0.4)
    shift = ceil(10 * maximum([std(tfom.tel.lm.s[inds[i], :] .* norm(tfom.tel.lm.M[:, inds[i]])) for i in inds])) / 2
    half_shift = ceil(shift) / 2
    for i in reverse(inds)
        my_scatter!(plt, times_nu, (tfom.tel.lm.s[i, :] .* norm(tfom.tel.lm.M[:, i])) .- (shift * (i - 1) + half_shift); label="Weights $i", color=plt_colors[((i - inds[1] + 3) % 19) + 1])
        hline!([-(shift * (i - 1) + half_shift)]; label="", color=plt_colors[i - inds[1] + 2], lw=3, alpha=0.4)
    end
    if display_plt; display(plt) end
    return plt
end

function status_plot(tfo::StellarSpectraObservationFitting.TFOutput, tfd::StellarSpectraObservationFitting.TFData; plot_epoch::Int=10, tracker::Int=0, display_plt::Bool=true)
    obs_λ = exp.(tfd.log_λ_obs[:, plot_epoch])
    plot_star_λs = exp.(tfd.log_λ_star[:, plot_epoch])
    plt = plot_spectrum(; legend = :bottomright, layout = grid(2, 1, heights=[0.85, 0.15]))

    plot!(plt[1], obs_λ, tfo.tel[:, plot_epoch], label="Telluric Model")

    shift = 1.1 - minimum(tfo.tel[:, plot_epoch])
    star_model = tfo.star[:, plot_epoch] + tfo.rv[:, plot_epoch]
    plot!(plt[1], obs_λ, star_model .- shift, label="Stellar Model")

    shift += 1.1 - minimum(star_model)
    my_scatter!(plt[1], obs_λ, tfd.flux[:, plot_epoch] .- shift, label="Observed Data", color=:white, alpha=0.1, xlabel="")
    plot!(plt[1], obs_λ, tfo.tel[:, plot_epoch] + star_model .- (1 + shift), label="Full Model", ls=:dash, color=:white)

    my_scatter!(plt[2], obs_λ, tfd.flux[:, plot_epoch] - (tfo.tel[:, plot_epoch] + tfo.star[:, plot_epoch] + tfo.rv[:, plot_epoch] .- 1), ylabel="Residuals", label="", alpha=0.1, color=:white)
    if display_plt; display(plt) end
    return plt
end

function separate_status_plot(tfo::StellarSpectraObservationFitting.TFOutput, tfd::StellarSpectraObservationFitting.TFData; plot_epoch::Int=10, tracker::Int=0, display_plt::Bool=true)
    obs_λ = exp.(tfd.log_λ_obs[:, plot_epoch])
    plot_star_λs = exp.(tfd.log_λ_star[:, plot_epoch])
    plt = plot_spectrum(; legend = :bottomleft, size=(800,800))
    plot!(plt[1], obs_λ, tfd.flux[:, plot_epoch] ./ (tfo.star[:, plot_epoch] + tfo.rv[:, plot_epoch]), label="predicted tel", alpha = 0.5, title="Tellurics")
    plot!(plt[1], obs_λ, tfo.tel[:, plot_epoch], label="model tel: $tracker", alpha = 0.5)
    plot!(plt[2], plot_star_λs, tfd.flux[:, plot_epoch] ./ tfo.tel[:, plot_epoch], label="predicted star", alpha = 0.5)
    plot!(plt[2], plot_star_λs, tfo.star[:, plot_epoch] + tfo.rv[:, plot_epoch], label="model star: $tracker", alpha = 0.5)
    if display_plt; display(plt) end
    return plt
end
