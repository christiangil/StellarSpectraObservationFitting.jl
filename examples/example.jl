## Setup
using Pkg
Pkg.activate("examples")
Pkg.instantiate()

using JLD2
import telfitting
# Pkg.status("telfitting")
# @time include("C:/Users/chris/Dropbox/GP_research/julia/telfitting/src/telfitting.jl")
tf = telfitting

## Loading (pregenerated) data

# include("data_structs.jl")
# @load "C:/Users/chris/OneDrive/Desktop/telfitting/telfitting_workspace_smol_150k.jld2" Spectra airmasses obs_resolution obs_λ planet_P_nu rvs_activ_no_noise rvs_activ_noisy rvs_kep_nu times_nu plot_times plot_rvs_kep true_tels
# @load "C:/Users/chris/OneDrive/Desktop/telfitting/telfitting_workspace_150k.jld2" quiet λ_nu true_tels_mean

## Setting up necessary variables and functions

@load "C:/Users/chris/OneDrive/Desktop/telfitting/tf_model_150k" tf_model n_obs len_obs tf_data star_model_res tel_model_res

using StatsBase

n_obs_train = Int(round(0.75 * n_obs))
testing_inds = sort(sample(1:n_obs, n_obs-n_obs_train; replace=false))
training_inds = [i for i in 1:n_obs if !(i in testing_inds)]

tf_data_train = tf.TFData(tf_data, training_inds)
tf_data_test = tf.TFData(tf_data, testing_inds)

tf_model_train = tf_model(training_inds)
tf_model_test = tf_model(testing_inds)

_loss(tel, star, rv, tfd) =
    sum((((tel .* (star + rv)) - flux_obs) .^ 2) ./ var_obs)
_loss_tel(star, rv, flux_obs, var_obs) = _loss(tf.tel_model(tf_model), star, rv, flux_obs, var_obs) + tf.tel_prior(tf_model)
_loss_star(tel, rv, flux_obs, var_obs) = _loss(tel, tf.star_model(tf_model), rv, flux_obs, var_obs) + tf.star_prior(tf_model)
_loss_rv(tel, star, flux_obs, var_obs) = _loss(tel, star, tf.rv_model(tf_model), flux_obs, var_obs)
loss() = _loss(tf.tel_model(tf_model), tf.star_model(tf_model), tf.rv_model(tf_model), flux_obs, var_obs)
loss_tel() = _loss_tel(tf.star_model(tf_model), tf.rv_model(tf_model), flux_obs, var_obs)
loss_star() = _loss_star(tf.tel_model(tf_model), tf.rv_model(tf_model), flux_obs, var_obs)
loss_rv() = _loss_rv(tf.tel_model(tf_model), tf.star_model(tf_model), flux_obs, var_obs)

using Plots
# plot(λ_nu, quiet; label="SOAP", xrange=(610, 670))
# plot!(tfm.star.λ, tf_model.star.lm.μ[:]; label="template")
#
# plot(obs_λ, true_tels_mean; label="true tels")
# plot!(tf_model.tel.λ, tf_model.tel.lm.μ[:]; label="template")
# plot!(tf_model.tel.λ, tf_model.tel.lm.M[:, 1]; label="M 1")
# plot!(tf_model.tel.λ, tf_model.tel.lm.M[:, 2]; label="M 2")

using Flux, Zygote, Optim

θ_tel = params(tf_model.tel.lm.M, tf_model.tel.lm.s, tf_model.tel.lm.μ)
θ_star = params(tf_model.star.lm.M, tf_model.star.lm.s, tf_model.star.lm.μ)
θ_rv = params(tf_model.rv.lm.s)
f_tel, g_tel, fg_tel!, p0_tel = tf.optfuns(loss_tel, θ_tel)
f_star, g_star, fg_star!, p0_star = tf.optfuns(loss_star, θ_star)
f_rv, g_rv, fg_rv!, p0_rv = tf.optfuns(loss_rv, θ_rv)

resid_stds = [std((rvs_notel - rvs_kep_nu) - rvs_activ_no_noise)]
losses = [loss()]
tracker = 0

plot_spectrum(; kwargs...) = plot(; xlabel = "Wavelength (nm)", ylabel = "Continuum Normalized Flux", dpi = 400, kwargs...)
plot_rv(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "RV (m/s)", dpi = 400, kwargs...)
function status_plot(; plot_epoch=1)
    tel_model_result = tf.tel_model(tf_model)
    star_model_result = tf.star_model(tf_model)
    rv_model_result = tf.rv_model(tf_model)

    l = @layout [a; b]
    # predict_plot = plot_spectrum(; legend = :bottomleft, size=(800,1200), layout = l)
    # predict_plot = plot_spectrum(; xlim=(627.8,628.3), legend=:bottomleft, size=(800,1200), layout = l) # o2
    # predict_plot = plot_spectrum(; xlim=(651.5,652), legend=:bottomleft, size=(800,1200), layout = l)  # h2o
    predict_plot = plot_spectrum(; xlim = (647, 656), legend = :bottomleft, size=(800,1200), layout = l)  # h2o
    plot!(predict_plot[1], obs_λ, true_tels[:, plot_epoch], label="true tel")
    plot!(predict_plot[1], obs_λ, flux_obs[:, plot_epoch] ./ (star_model_result[:, plot_epoch] + rv_model_result[:, plot_epoch]), label="predicted tel", alpha = 0.5)
    plot!(predict_plot[1], obs_λ, tel_model_result[:, plot_epoch], label="model tel: $tracker", alpha = 0.5)
    plot_star_λs = exp.(Spectra[plot_epoch].log_λ_bary)
    plot!(predict_plot[2], plot_star_λs, flux_obs[:, plot_epoch] ./ true_tels[:, plot_epoch], label="true star", )
    plot!(predict_plot[2], plot_star_λs, flux_obs[:, plot_epoch] ./ tel_model_result[:, plot_epoch], label="predicted star", alpha = 0.5)
    plot!(predict_plot[2], plot_star_λs, star_model_result[:, plot_epoch] + rv_model_result[:, plot_epoch], label="model star: $tracker", alpha = 0.5)
    display(predict_plot)
end
plot_epoch = 60
status_plot(; plot_epoch=plot_epoch)

OOptions = Optim.Options(iterations=10, f_tol=1e-3, g_tol=1e5)

# Optim.optimize(Optim.only_fg!(fg_ts!), p0_ts, OOptions)
# tf_model.rv.lm.M[:] = tf.calc_doppler_component_RVSKL(tf_model.star.λ, tf_model.star.lm.μ)
# Optim.optimize(Optim.only_fg!(fg_rv!), p0_rv, OOptions)

println("guess $tracker, std=$(round(std(rvs_notel - rvs_kep_nu - rvs_activ_no_noise), digits=5))")
rvs_notel_opt = copy(rvs_notel)
light_speed_nu = 299792458
@time for i in 1:4

    # optimize star
    tf.Flux_optimize!(fg_star!, p0_star, θ_star, OOptions)

    # optimize tellurics
    tf.Flux_optimize!(fg_tel!, p0_tel, θ_tel, OOptions)

    # optimize RVs
    tf_model.rv.lm.M[:] = tf.calc_doppler_component_RVSKL(tf_model.star.λ, tf_model.star.lm.μ)
    tf.Flux_optimize!(fg_rv!, p0_rv, θ_rv, OOptions)
    rvs_notel_opt[:] = (tf_model.rv.lm.s .* light_speed_nu)'

    append!(resid_stds, [std(rvs_notel_opt - rvs_kep_nu - rvs_activ_no_noise)])
    append!(losses, [loss()])

    status_plot(; plot_epoch=plot_epoch)
    tracker += 1
    println("guess $tracker")
    println("loss   = $(losses[end])")
    println("rv std = $(round(std((rvs_notel_opt - rvs_kep_nu) - rvs_activ_no_noise), digits=5))")
end

plot(resid_stds; xlabel="iter", ylabel="predicted RV - active RV RMS", legend=false)
plot(losses; xlabel="iter", ylabel="loss", legend=false)

## Plots

plot_stuff = true

if plot_stuff

    # Compare first guess at RVs to true signal
    predict_plot = plot_rv()
    plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
    png(predict_plot, "examples/figs/model_0_phase.png")

    # Compare RV differences to actual RVs from activity
    predict_plot = plot_rv()
    plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
    plot!(predict_plot, times_nu, rvs_naive - rvs_kep_nu, st=:scatter, ms=3, color=:blue, label="Before model")
    png(predict_plot, "examples/figs/model_0.png")

    # Compare second guess at RVs to true signal
    predict_plot = plot_rv()
    plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    png(predict_plot, "examples/figs/model_1_phase.png")

    # Compare RV differences to actual RVs from activity
    predict_plot = plot_rv()
    plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
    plot!(predict_plot, times_nu, rvs_notel - rvs_kep_nu, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    png(predict_plot, "examples/figs/model_1.png")

    # Compare second guess at RVs to true signal
    predict_plot = plot_rv()
    plot!(predict_plot, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    plot!(predict_plot, times_nu .% planet_P_nu, rvs_notel_opt, st=:scatter, ms=3, color=:darkgreen, label="After optimization")
    png(predict_plot, "examples/figs/model_2_phase.png")

    # Compare RV differences to actual RVs from activity
    predict_plot = plot_rv()
    plot!(predict_plot, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
    plot!(predict_plot, times_nu, rvs_notel - rvs_kep_nu, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    plot!(predict_plot, times_nu, rvs_notel_opt - rvs_kep_nu, st=:scatter, ms=3, color=:darkgreen, label="After optimization")
    png(predict_plot, "examples/figs/model_2.png")

    # predict_plot = plot_spectrum(; xlim=(627.8,628.3)) # o2
    # predict_plot = plot_spectrum(; xlim = (647, 656))  # h2o
    # predict_plot = plot_spectrum(; xlim=(651.5,652))  # h2o
    predict_plot = plot_spectrum(; title="Stellar model")
    plot!(tf_model.star.λ, tf_model.star.lm.M[:, 1]; label="basis 1")
    plot!(tf_model.star.λ, tf_model.star.lm.M[:, 2]; label="basis 2")
    plot!(tf_model.star.λ, tf_model.star.lm.μ; label="μ")
    png(predict_plot, "examples/figs/model_star_basis.png")

    predict_plot = plot_spectrum(; xlim=(647, 656), title="Stellar model")  # h2o
    plot!(tf_model.star.λ, tf_model.star.lm.M[:, 1]; label="basis 1")
    plot!(tf_model.star.λ, tf_model.star.lm.M[:, 2]; label="basis 2")
    plot!(tf_model.star.λ, tf_model.star.lm.μ; label="μ star")
    plot!(tf_model.tel.λ, tf_model.tel.lm.μ; label="μ tel")
    png(predict_plot, "examples/figs/model_star_basis_h2o.png")

    plot_scores(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "Weights", dpi = 400, kwargs...)
    predict_plot = plot_scores(; title="Stellar model")
    scatter!(times_nu, tf_model.star.lm.s[1, :]; label="weights 1")
    scatter!(times_nu, tf_model.star.lm.s[2, :]; label="weights 2")
    png(predict_plot, "examples/figs/model_star_weights.png")

    predict_plot = plot_spectrum(; title="Telluric model")
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:, 1]; label="basis 1")
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:,2]; label="basis 2")
    plot!(tf_model.tel.λ, tf_model.tel.lm.μ; label="μ")
    png(predict_plot, "examples/figs/model_tel_basis.png")

    predict_plot = plot_spectrum(;xlim=(647, 656), title="Telluric model (H2O)")  # h2o
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:,1]; label="basis 1")
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:,2]; label="basis 2")
    plot!(tf_model.tel.λ, tf_model.tel.lm.μ; label="μ")
    png(predict_plot, "examples/figs/model_tel_basis_h2o.png")
    predict_plot = plot_spectrum(;xlim=(627.8,628.3), title="Telluric model (O2)")  # o2
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:,1]; label="basis 1")
    plot!(tf_model.tel.λ, tf_model.tel.lm.M[:,2]; label="basis 2")
    plot!(tf_model.tel.λ, tf_model.tel.lm.μ; label="μ")
    png(predict_plot, "examples/figs/model_tel_basis_o2.png")

    predict_plot = plot_scores(; title="Telluric model")
    scatter!(times_nu, tf_model.tel.lm.s[1, :]; label="weights 1")
    scatter!(times_nu, tf_model.tel.lm.s[2, :]; label="weights 2")
    scatter!(times_nu, airmasses .- 3; label="airmasses")
    png(predict_plot, "examples/figs/model_tel_weights.png")
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
