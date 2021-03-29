## Setup
using Pkg
Pkg.activate("examples")
# Pkg.add("JLD2")
# Pkg.add("UnitfulAstro")
# Pkg.add("Unitful")
# Pkg.develop(;path="C:/Users/chris/Dropbox/GP_research/julia/telfitting")
# Pkg.add("Stheno")
# Pkg.add("TemporalGPs")
# Pkg.add("Distributions")
# Pkg.add("Plots")
Pkg.instantiate()

using JLD2
import telfitting; tf = telfitting

## Loading (pregenerated) data

include("data_structs.jl")
@load "E:/telfitting/telfitting_workspace_smol_150k.jld2" Spectra obs_resolution

## Setting up necessary variables and functions

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
tf_data = tf.TFData(flux_obs, var_obs, log_λ_obs, log_λ_star)

## Initializing models

star_model_res = 2 * sqrt(2) * obs_resolution
tel_model_res = sqrt(2) * obs_resolution

@time tf_model = tf.TFModel(tf_data, star_model_res, tel_model_res)

@time rvs_notel, rvs_naive = tf.initialize!(tf_model, tf_data; use_gp=true)

@save "E:/telfitting/tf_model_150k.jld2" tf_model n_obs tf_data rvs_notel rvs_naive
