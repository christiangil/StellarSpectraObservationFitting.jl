## Setup
using Pkg
Pkg.activate("examples")
# Pkg.add("JLD2")
# Pkg.add("UnitfulAstro")
# Pkg.add("Unitful")
# Pkg.add(;path="C:/Users/chris/Dropbox/GP_research/julia/telfitting")
# Pkg.add("Stheno")
# Pkg.add("TemporalGPs")
# Pkg.add("Distributions")
# Pkg.add("Plots")
Pkg.instantiate()

using JLD2
# using Stheno
# using TemporalGPs
using UnitfulAstro, Unitful
@time include("C:/Users/chris/Dropbox/GP_research/julia/telfitting/src/telfitting.jl")
tf = Main.telfitting

## Loading (pregenerated) data

include("data_structs.jl")
@load "C:/Users/chris/OneDrive/Desktop/telfitting/telfitting_workspace_smol_150k.jld2" Spectra airmasses obs_resolution obs_λ planet_P_nu rvs_activ_no_noise rvs_activ_noisy rvs_kep_nu times_nu plot_times plot_rvs_kep true_tels
@load "C:/Users/chris/OneDrive/Desktop/telfitting/telfitting_workspace_150k.jld2" quiet λ_nu true_tels_mean

## Setting up necessary variables and functions

light_speed_nu = 299792458
plot_stuff = true

n_obs = length(Spectra)
len_obs = length(Spectra[1].log_λ_obs)
flux_obs = ones(len_obs, n_obs)
var_obs = zeros(len_obs, n_obs)
log_λ_obs = zeros(len_obs, n_obs)
log_λ_star = zeros(len_obs, n_obs)
for i in 1:n_obs # 13s
    flux_obs[:, i] = Spectra[i].flux_obs
    var_obs[:, i] = Spectra[i].var_obs
    log_λ_obs[:, i] = Spectra[i].log_λ_obs
    log_λ_star[:, i] = Spectra[i].log_λ_bary
end

## Initializing models

star_model_res = 2 * sqrt(2) * obs_resolution
tel_model_res = sqrt(2) * obs_resolution

@time tf_model = tf.TFModel(log_λ_obs, log_λ_star, star_model_res, tel_model_res)

@time rvs_notel, rvs_naive = tf.initialize!(tf_model, flux_obs, var_obs, log_λ_obs, log_λ_star; use_gp=true)

@save "C:/Users/chris/OneDrive/Desktop/telfitting/tf_model_150k" tf_model n_obs len_obs flux_obs var_obs log_λ_obs log_λ_star star_model_res tel_model_res
