## Setup
using Pkg
Pkg.activate("examples")
Pkg.instantiate()

using JLD2
using Statistics
import telfitting; tf = telfitting

plot_stuff = true
use_telstar = true
improve_regularization = false

## Setting up necessary variables and functions

@load "C:/Users/chris/OneDrive/Desktop/telfitting/tf_model_150k.jld2" tf_model n_obs tf_data rvs_notel rvs_naive

if improve_regularization
    using StatsBase
    n_obs_train = Int(round(0.75 * n_obs))
    training_inds = sort(sample(1:n_obs, n_obs_train; replace=false))
    tf.fit_regularization!(tf_model, tf_data, training_inds; use_telstar=use_telstar)
else
    tf_model.reg_tel[:L2_μ] = 1e7
    tf_model.reg_tel[:L1_μ] = 1e3
    tf_model.reg_tel[:L1_μ₊_factor] = 2
    tf_model.reg_tel[:L2_M] = 1e7
    tf_model.reg_tel[:L1_M] = 1e5
    tf_model.reg_tel[:shared_M] = 0.

    tf_model.reg_star[:L2_μ] = 1e6
    tf_model.reg_star[:L1_μ] = 1e-1
    tf_model.reg_star[:L1_μ₊_factor] = 2
    tf_model.reg_star[:L2_M] = 1e11
    tf_model.reg_star[:L1_M] = 1e7
    tf_model.reg_star[:shared_M] = 0.
end

tf_output = tf.TFOutput(tf_model)

if use_telstar
    tf_workspace, loss = tf.TFWorkspaceTelStar(tf_model, tf_output, tf_data; return_loss_f=true)
else
    tf_workspace, loss = tf.TFWorkspace(tf_model, tf_output, tf_data; return_loss_f=true)
end

using Plots
if plot_stuff
    @load "C:/Users/chris/OneDrive/Desktop/telfitting/telfitting_workspace_smol_150k.jld2" airmasses planet_P_nu rvs_activ_no_noise rvs_activ_noisy rvs_kep_nu times_nu plot_times plot_rvs_kep true_tels

    plot_spectrum(; kwargs...) = plot(; xlabel = "Wavelength (nm)", ylabel = "Continuum Normalized Flux", dpi = 400, kwargs...)
    plot_rv(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "RV (m/s)", dpi = 400, kwargs...)

    function status_plot(tfo::tf.TFOutput, tfd::tf.TFData; plot_epoch::Int=10, tracker::Int=0)
        obs_λ = exp.(tfd.log_λ_obs[:, plot_epoch])
        l = @layout [a; b]
        # predict_plot = plot_spectrum(; legend = :bottomleft, size=(800,1200), layout = l)
        # predict_plot = plot_spectrum(; xlim=(627.8,628.3), legend=:bottomleft, size=(800,1200), layout = l) # o2
        # predict_plot = plot_spectrum(; xlim=(651.5,652), legend=:bottomleft, size=(800,1200), layout = l)  # h2o
        predict_plot = plot_spectrum(; xlim = (647, 656), legend = :bottomleft, size=(800,1200), layout = l)  # h2o
        plot!(predict_plot[1], obs_λ, true_tels[:, plot_epoch], label="true tel")
        plot!(predict_plot[1], obs_λ, tfd.flux[:, plot_epoch] ./ (tfo.star[:, plot_epoch] + tfo.rv[:, plot_epoch]), label="predicted tel", alpha = 0.5)
        plot!(predict_plot[1], obs_λ, tfo.tel[:, plot_epoch], label="model tel: $tracker", alpha = 0.5)
        plot_star_λs = exp.(tfd.log_λ_star[:, plot_epoch])
        plot!(predict_plot[2], plot_star_λs, tfd.flux[:, plot_epoch] ./ true_tels[:, plot_epoch], label="true star")
        plot!(predict_plot[2], plot_star_λs, tfd.flux[:, plot_epoch] ./ tfo.tel[:, plot_epoch], label="predicted star", alpha = 0.5)
        plot!(predict_plot[2], plot_star_λs, tfo.star[:, plot_epoch] + tfo.rv[:, plot_epoch], label="model star: $tracker", alpha = 0.5)
        display(predict_plot)
    end
    status_plot(tf_output, tf_data)
end

light_speed_nu = 299792458
rvs_notel = (tf_model.rv.lm.s .* light_speed_nu)'
rvs_std(rvs; inds=:) = std((rvs - rvs_kep_nu[inds]) - rvs_activ_no_noise[inds])
resid_stds = [rvs_std(rvs_notel)]
losses = [loss()]
tracker = 0
println("guess $tracker, std=$(round(rvs_std(rvs_notel), digits=5))")
rvs_notel_opt = copy(rvs_notel)

@time for i in 1:8
    tf.train_TFModel!(tf_workspace)
    rvs_notel_opt[:] = (tf_model.rv.lm.s .* light_speed_nu)'

    append!(resid_stds, [rvs_std(rvs_notel_opt)])
    append!(losses, [loss()])

    status_plot(tf_output, tf_data)
    tracker += 1
    println("guess $tracker")
    println("loss   = $(losses[end])")
    println("rv std = $(round(rvs_std(rvs_notel_opt), digits=5))")
end

plot(0:tracker, resid_stds; xlabel="iter", ylabel="predicted RV - active RV RMS", legend=false)
plot(0:tracker, losses; xlabel="iter", ylabel="loss", legend=false)

## Plots

using LinearAlgebra

if plot_stuff

    fig_dir = "examples/figs/"

    # Compare first guess at RVs to true signal
    predict_plot = plot_rv()
    plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
    png(predict_plot, fig_dir * "model_0_phase.png")

    # Compare RV differences to actual RVs from activity
    predict_plot = plot_rv()
    plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
    plot!(predict_plot, times_nu, rvs_naive - rvs_kep_nu, st=:scatter, ms=3, color=:blue, label="Before model")
    png(predict_plot, fig_dir * "model_0.png")

    # Compare second guess at RVs to true signal
    predict_plot = plot_rv()
    plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    png(predict_plot, fig_dir * "model_1_phase.png")

    # Compare RV differences to actual RVs from activity
    predict_plot = plot_rv()
    plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
    plot!(predict_plot, times_nu, rvs_notel - rvs_kep_nu, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    png(predict_plot, fig_dir * "model_1.png")

    # Compare second guess at RVs to true signal
    predict_plot = plot_rv()
    plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel_opt, st=:scatter, ms=3, color=:darkgreen, label="After optimization")
    png(predict_plot, fig_dir * "model_2_phase.png")

    # Compare RV differences to actual RVs from activity
    predict_plot = plot_rv()
    plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
    plot!(predict_plot, times_nu, rvs_notel - rvs_kep_nu, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    plot!(predict_plot, times_nu, rvs_notel_opt - rvs_kep_nu, st=:scatter, ms=3, color=:darkgreen, label="After optimization")
    png(predict_plot, fig_dir * "model_2.png")

    # predict_plot = plot_spectrum(; xlim=(627.8,628.3)) # o2
    # predict_plot = plot_spectrum(; xlim = (647, 656))  # h2o
    # predict_plot = plot_spectrum(; xlim=(651.5,652))  # h2o
    predict_plot = plot_spectrum(; title="Stellar model")
    plot!(tf_model.star.λ, tf_model.star.lm.M[:, 1] ./ norm(tf_model.star.lm.M[:, 1]); label="basis 1")
    plot!(tf_model.star.λ, tf_model.star.lm.M[:, 2] ./ norm(tf_model.star.lm.M[:, 2]); label="basis 2")
    plot!(tf_model.star.λ, tf_model.star.lm.μ; label="μ")
    png(predict_plot, fig_dir * "model_star_basis.png")

    predict_plot = plot_spectrum(; xlim=(647, 656), title="Stellar model")  # h2o
    plot!(tf_model.star.λ, tf_model.star.lm.M[:, 1] ./ norm(tf_model.star.lm.M[:, 1]); label="basis 1")
    plot!(tf_model.star.λ, tf_model.star.lm.M[:, 2] ./ norm(tf_model.star.lm.M[:, 2]); label="basis 2")
    plot!(tf_model.star.λ, tf_model.star.lm.μ; label="μ star")
    plot!(tf_model.tel.λ, tf_model.tel.lm.μ; label="μ tel")
    png(predict_plot, fig_dir * "model_star_basis_h2o.png")

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

    predict_plot = plot_spectrum(;xlim=(647, 656), title="Telluric model (H2O)")  # h2o
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:,1] ./ norm(tf_model.tel.lm.M[:, 1]); label="basis 1")
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:,2] ./ norm(tf_model.tel.lm.M[:, 2]); label="basis 2")
    plot!(tf_model.tel.λ, tf_model.tel.lm.μ; label="μ")
    png(predict_plot, fig_dir * "model_tel_basis_h2o.png")
    predict_plot = plot_spectrum(;xlim=(627.8,628.3), title="Telluric model (O2)")  # o2
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:,1] ./ norm(tf_model.tel.lm.M[:, 1]); label="basis 1")
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:,2] ./ norm(tf_model.tel.lm.M[:, 2]); label="basis 2")
    plot!(tf_model.tel.λ, tf_model.tel.lm.μ; label="μ")
    png(predict_plot, fig_dir * "model_tel_basis_o2.png")

    predict_plot = plot_scores(; title="Telluric model")
    scatter!(times_nu, tf_model.tel.lm.s[1, :] .* norm(tf_model.tel.lm.M[:, 1]); label="weights 1")
    scatter!(times_nu, tf_model.tel.lm.s[2, :] .* norm(tf_model.tel.lm.M[:, 2]); label="weights 2")
    scatter!(times_nu, airmasses; label="airmasses")
    png(predict_plot, fig_dir * "model_tel_weights.png")
end

std((rvs_naive - rvs_kep_nu) - rvs_activ_no_noise)
std((rvs_notel - rvs_kep_nu) - rvs_activ_no_noise)
std((rvs_notel_opt - rvs_kep_nu) - rvs_activ_no_noise)
std((rvs_naive - rvs_kep_nu) - rvs_activ_noisy)
std((rvs_notel - rvs_kep_nu) - rvs_activ_noisy)
std((rvs_notel_opt - rvs_kep_nu) - rvs_activ_noisy)
std(rvs_activ_noisy - rvs_activ_no_noise)  # best case
std(rvs_notel_opt - rvs_kep_nu)
std(rvs_activ_noisy)
std(rvs_activ_no_noise)
