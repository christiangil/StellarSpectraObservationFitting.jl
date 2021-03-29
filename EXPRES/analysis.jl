## Setup
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

using JLD2
using Statistics
import telfitting; tf = telfitting

stars = ["10700", "26965"]
star = stars[1]
plot_stuff = true
use_telstar = true
improve_regularization = true

## Setting up necessary variables and functions

@load "C:/Users/chris/OneDrive/Desktop/telfitting/" * star * ".jld2" tf_model n_obs tf_data rvs_notel

tf_output = tf.TFOutput(tf_model)

if use_telstar
    tf_workspace, loss = tf.TFWorkspaceTelStar(tf_model, tf_output, tf_data; return_loss_f=true)
else
    tf_workspace, loss = tf.TFWorkspace(tf_model, tf_output, tf_data; return_loss_f=true)
end

using Plots
if plot_stuff
    @load "C:/Users/chris/OneDrive/Desktop/telfitting/" * star * ".jld2" rvs_naive airmasses times_nu

    plot_spectrum(; kwargs...) = plot(; xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux", dpi = 400, kwargs...)
    plot_rv(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "RV (m/s)", dpi = 400, kwargs...)

    function status_plot(tfo::tf.TFOutput, tfd::tf.TFData; plot_epoch::Int=10, tracker::Int=0)
        obs_λ = exp.(tfd.log_λ_obs[:, plot_epoch])
        l = @layout [a; b]
        predict_plot = plot_spectrum(; legend = :bottomleft, size=(800,1200), layout = l)
        plot!(predict_plot[1], obs_λ, tfd.flux[:, plot_epoch] ./ (tfo.star[:, plot_epoch] + tfo.rv[:, plot_epoch]), label="predicted tel", alpha = 0.5)
        plot!(predict_plot[1], obs_λ, tfo.tel[:, plot_epoch], label="model tel: $tracker", alpha = 0.5)
        plot_star_λs = exp.(tfd.log_λ_star[:, plot_epoch])
        plot!(predict_plot[2], plot_star_λs, tfd.flux[:, plot_epoch] ./ tfo.tel[:, plot_epoch], label="predicted star", alpha = 0.5)
        plot!(predict_plot[2], plot_star_λs, tfo.star[:, plot_epoch] + tfo.rv[:, plot_epoch], label="model star: $tracker", alpha = 0.5)
        display(predict_plot)
    end
    status_plot(tf_output, tf_data)
end

light_speed_nu = 299792458
rvs_notel = (tf_model.rv.lm.s .* light_speed_nu)'
resid_stds = [std(rvs_notel)]
losses = [loss()]
tracker = 0
println("guess $tracker, std=$(round(std(rvs_notel), digits=5))")
rvs_notel_opt = copy(rvs_notel)

if improve_regularization
    using StatsBase
    n_obs_train = Int(round(0.75 * n_obs))
    training_inds = sort(sample(1:n_obs, n_obs_train; replace=false))
    tf.fit_regularization!(tf_model, tf_data, training_inds; use_telstar=use_telstar)
    println(tf_model.reg_tel)
    println(tf_model.reg_star)
elseif use_telstar
    tf_model.reg_tel[:L2_μ] = 1e8
    tf_model.reg_tel[:L1_μ] = 1e5
    tf_model.reg_tel[:L1_μ₊_factor] = 8.6
    delete!(tf_model.reg_tel, :shared_M)
    tf_model.reg_tel[:L2_M] = 1e9
    tf_model.reg_tel[:L1_M] = 1e6

    tf_model.reg_star[:L2_μ] = 1e5
    tf_model.reg_star[:L1_μ] = 1e5
    tf_model.reg_star[:L1_μ₊_factor] = 8.6
    delete!(tf_model.reg_star, :shared_M)
    tf_model.reg_star[:L2_M] = 1e8
    tf_model.reg_star[:L1_M] = 1e9
end

@time for i in 1:8
    tf.train_TFModel!(tf_workspace)
    rvs_notel_opt[:] = (tf_model.rv.lm.s .* light_speed_nu)'

    append!(resid_stds, [std(rvs_notel_opt)])
    append!(losses, [loss()])

    status_plot(tf_output, tf_data)
    tracker += 1
    println("guess $tracker")
    println("loss   = $(losses[end])")
    println("rv std = $(round(std(rvs_notel_opt), digits=5))")
end

plot(0:tracker, resid_stds; xlabel="iter", ylabel="predicted RV - active RV RMS", legend=false)
plot(0:tracker, losses; xlabel="iter", ylabel="loss", legend=false)

## Plots
using LinearAlgebra

if plot_stuff

    fig_dir = "EXPRES/figs/" * star * "_"

    # # Compare RV differences to actual RVs from activity
    # predict_plot = plot_rv()
    # plot!(predict_plot, times_nu, rvs_naive, st=:scatter, ms=3, color=:red, label="Naive")
    # plot!(predict_plot, times_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    # png(predict_plot, fig_dir * "model_1.png")

    # Compare RV differences to actual RVs from activity
    predict_plot = plot_rv()
    plot!(predict_plot, times_nu, rvs_naive, st=:scatter, ms=3, color=:red, label="Naive, std: $(round(std(rvs_naive), digits=3))")
    plot!(predict_plot, times_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization, std: $(round(std(rvs_notel), digits=3))")
    plot!(predict_plot, times_nu, rvs_notel_opt, st=:scatter, ms=3, color=:darkgreen, label="After optimization, std: $(round(std(rvs_notel_opt), digits=3))")
    png(predict_plot, fig_dir * "model_2.png")

    predict_plot = plot_spectrum(; title="Stellar model")
    plot!(tf_model.star.λ, tf_model.star.lm.M[:, 1] ./ norm(tf_model.star.lm.M[:, 1]); label="basis 1")
    plot!(tf_model.star.λ, tf_model.star.lm.M[:, 2] ./ norm(tf_model.star.lm.M[:, 2]); label="basis 2")
    plot!(tf_model.star.λ, tf_model.star.lm.μ; label="μ")
    png(predict_plot, fig_dir * "model_star_basis.png")

    plot_scores(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "Weights", dpi = 400, kwargs...)
    predict_plot = plot_scores(; title="Stellar model")
    scatter!(times_nu, tf_model.star.lm.s[1, :] .* norm(tf_model.star.lm.M[:, 1]); label="weights 1")
    scatter!(times_nu, tf_model.star.lm.s[2, :] .* norm(tf_model.star.lm.M[:, 2]); label="weights 2")
    png(predict_plot, fig_dir * "model_star_weights.png")

    predict_plot = plot_spectrum(; title="Telluric model")
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:, 1] ./ norm(tf_model.tel.lm.M[:, 1]); label="basis 1")
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:, 2] ./ norm(tf_model.tel.lm.M[:, 2]); label="basis 2")
    plot!(tf_model.tel.λ, tf_model.tel.lm.μ; label="μ")
    png(predict_plot, fig_dir * "model_tel_basis.png")

    predict_plot = plot_scores(; title="Telluric model")
    scatter!(times_nu, tf_model.tel.lm.s[1, :] .* norm(tf_model.tel.lm.M[:, 1]); label="weights 1")
    scatter!(times_nu, tf_model.tel.lm.s[2, :] .* norm(tf_model.tel.lm.M[:, 2]); label="weights 2")
    scatter!(times_nu, airmasses; label="airmasses")
    png(predict_plot, fig_dir * "model_tel_weights.png")
end

std(rvs_naive)
std(rvs_notel)
std(rvs_notel_opt)
