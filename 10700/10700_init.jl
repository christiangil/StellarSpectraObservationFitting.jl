## Setup
using Pkg
Pkg.activate("10700")
Pkg.instantiate()

## Importing data with Eric's code

# Pkg.develop(; url="https://github.com/RvSpectML/EchelleInstruments.jl")
# Pkg.develop(; url="https://github.com/RvSpectML/RvSpectML.jl")
# Pkg.develop(; url="https://github.com/RvSpectML/RvSpectMLBase.jl")
# Pkg.develop(;path="C:/Users/chris/Dropbox/GP_research/julia/telfitting")

using RvSpectMLBase, RvSpectML
using EchelleInstruments, EchelleInstruments.EXPRES
using CSV, DataFrames, Query, StatsBase, Statistics, Dates

target_subdir = "10700/"   # USER: Replace with directory of your choice
fits_target_str = "10700"
paths_to_search_for_param = ["10700"]

expres_data_path = "C:/Users/chris/OneDrive/Desktop/"

# NOTE: make_manifest does not update its paths_to_search when default_paths_to_search is defined here, so if you change the line above, you must also include "paths_to_search=default_paths_to_search" in the make_manifest() function call below
pipeline_plan = PipelinePlan()
dont_make_plot!(pipeline_plan, :movie)
reset_all_needs!(pipeline_plan)
if need_to(pipeline_plan,:read_spectra)
    df_files = make_manifest(expres_data_path, target_subdir, EXPRES)
    # Reading in customized parameters from param.jl.
    eval(code_to_include_param_jl(paths_to_search=paths_to_search_for_param))
    # Reading in FITS files
    @time all_spectra = map(row->EXPRES.read_data(row; store_min_data=true, store_tellurics=true, normalization=:continuum, return_λ_obs=true),eachrow(df_files_use))
    #@time all_spectra = map(row->EXPRES.read_data(row,store_min_data=true, store_tellurics=true, store_blaze=true, store_continuum=true, store_pixel_mask=true,normalization=:continuum),eachrow(df_files_use))
    GC.gc()
    dont_need_to!(pipeline_plan,:read_spectra)
end

# Something buggy if include order 1.  Order 86 essentially ignored due to tellurics. First and last orders have NaN issues
#max_orders = 12:83
lfc_orders = 43:72
order_list_timeseries = extract_orders(all_spectra,pipeline_plan; orders_to_use=lfc_orders, recalc=true)

times_nu = pipeline_plan.cache[:extract_orders].times
airmasses = [parse(Float64, md[:airmass]) for md in pipeline_plan.cache[:extract_orders].metadata]

# @save "C:/Users/chris/OneDrive/Desktop/test.jld2" order_list_timeseries
# pipeline_plan.cache[:extract_orders] == order_list_timeseries

## Switching to my data format

using JLD2
import telfitting
tf = telfitting

obs_resolution = 150000
desired_order = 50

n_obs = length(all_spectra)
mask_inds = 770:6650

len_obs = length(mask_inds)
flux_obs = ones(len_obs, n_obs)
var_obs = zeros(len_obs, n_obs)
log_λ_obs = zeros(len_obs, n_obs)
log_λ_star = zeros(len_obs, n_obs)
for i in 1:n_obs # 13s
    flux_obs[:, i] = all_spectra[i].flux[mask_inds, desired_order]
    var_obs[:, i] = all_spectra[i].var[mask_inds, desired_order]
    log_λ_obs[:, i] = log.(all_spectra[i].λ_obs[mask_inds, desired_order])
    log_λ_star[:, i] = log.(all_spectra[i].λ[mask_inds, desired_order])
end
tf_data = tf.TFData(flux_obs, var_obs, log_λ_obs, log_λ_star)

## Initializing models

star_model_res = 2 * sqrt(2) * obs_resolution
tel_model_res = sqrt(2) * obs_resolution

@time tf_model = tf.TFModel(tf_data, star_model_res, tel_model_res)

@time rvs_notel, rvs_naive = tf.initialize!(tf_model, tf_data; use_gp=true)

@save "C:/Users/chris/OneDrive/Desktop/telfitting/10700.jld2" tf_model n_obs tf_data rvs_naive rvs_notel times_nu airmasses
