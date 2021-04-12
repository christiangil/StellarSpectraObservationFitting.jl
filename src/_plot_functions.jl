using LinearAlgebra
using Plots

plot_spectrum(; kwargs...) = plot(; xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux", dpi = 400, kwargs...)
plot_rv(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "RV (m/s)", dpi = 400, kwargs...)
plot_scores(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "Weights", dpi = 400, kwargs...)

function plot_model_rvs(times_nu::Vector{T}, rvs_naive::Vector{T}, rvs_notel::Vector{T}, rvs_notel_opt::Vector{T}) where {T<:Real}
    predict_plot = plot_rv()
    plot!(predict_plot, times_nu, rvs_naive, st=:scatter, ms=3, color=:red, label="Naive, std: $(round(std(rvs_naive), digits=3))")
    plot!(predict_plot, times_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization, std: $(round(std(rvs_notel), digits=3))")
    plot!(predict_plot, times_nu, rvs_notel_opt, st=:scatter, ms=3, color=:darkgreen, label="After optimization, std: $(round(std(rvs_notel_opt), digits=3))")
    display(predict_plot)
    return predict_plot
end

function plot_stellar_model_bases(tf_model::telfitting.TFOrderModel)
    predict_plot = plot_spectrum(; title="Stellar model bases")
    plot!(tf_model.star.λ, tf_model.star.lm.μ; label="μ")
    for i in 1:size(tf_model.star.lm.M, 2)
        plot!(tf_model.star.λ, tf_model.star.lm.M[:, i] ./ norm(tf_model.star.lm.M[:, i]); label="basis $i")
    end
    display(predict_plot)
    return predict_plot
end
function plot_stellar_model_scores(tf_model::telfitting.TFOrderModel)
    predict_plot = plot_scores(; title="Stellar model scores")
    for i in 1:size(tf_model.star.lm.M, 2)
        scatter!(times_nu, tf_model.star.lm.s[i, :] .* norm(tf_model.star.lm.M[:, i]); label="weights $i")
    end
    display(predict_plot)
    return predict_plot
end

function plot_telluric_model_bases(tf_model::telfitting.TFOrderModel)
    predict_plot = plot_spectrum(; title="Telluric model bases")
    plot!(tf_model.tel.λ, tf_model.tel.lm.μ; label="μ")
    for i in 1:size(tf_model.tel.lm.M, 2)
        plot!(tf_model.tel.λ, tf_model.tel.lm.M[:, i] ./ norm(tf_model.tel.lm.M[:, i]); label="basis $i")
    end
    display(predict_plot)
    return predict_plot
end
function plot_telluric_model_scores(tf_model::telfitting.TFOrderModel)
    predict_plot = plot_scores(; title="Telluric model scores")
    for i in 1:size(tf_model.tel.lm.M, 2)
        scatter!(times_nu, tf_model.tel.lm.s[i, :] .* norm(tf_model.tel.lm.M[:, i]); label="weights $i")
    end
    scatter!(times_nu, airmasses; label="airmasses")
    display(predict_plot)
    return predict_plot
end
