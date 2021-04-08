## Setup
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

## Importing data with Eric's code

stars = ["10700", "26965"]
star = stars[2]

using RvSpectMLBase, RvSpectML
using EchelleInstruments, EchelleInstruments.EXPRES
using CSV, DataFrames, Query, StatsBase, Statistics, Dates

target_subdir = star * "/"   # USER: Replace with directory of your choice
fits_target_str = star
paths_to_search_for_param = ["EXPRES"]

expres_data_path = "E:/telfitting/"

# NOTE: make_manifest does not update its paths_to_search when default_paths_to_search is defined here, so if you change the line above, you must also include "paths_to_search=default_paths_to_search" in the make_manifest() function call below
pipeline_plan = PipelinePlan()
dont_make_plot!(pipeline_plan, :movie)
reset_all_needs!(pipeline_plan)
masks = Array{UnitRange, 2}(undef, length(readdir(expres_data_path * target_subdir)), 86)
if need_to(pipeline_plan,:read_spectra)
    df_files = make_manifest(expres_data_path, target_subdir, EXPRES)
    # Reading in customized parameters from param.jl.
    eval(code_to_include_param_jl(paths_to_search=paths_to_search_for_param))
    # Reading in FITS files
    all_spectra = Spectra2DExtended[]
    for j in 1:size(masks, 1)
        row = eachrow(df_files_use)[j]
        spectra, mask = EXPRES.read_data(row; store_min_data=true, store_tellurics=true, normalization=:continuum, return_λ_obs=true, return_excalibur_mask=true)
        append!(all_spectra, [spectra])
        for i in 1:size(mask, 2)
            try
                x = findfirst(view(mask, :, i)):findlast(view(mask, :, i))
                masks[j, i] = x
                @assert all(mask[x, i] .== true)
            catch y
                if typeof(y)==MethodError
                    masks[j, i] = 0:0
                else
                    throw(y)
                end
            end
        end
    end
    GC.gc()
    dont_need_to!(pipeline_plan,:read_spectra)
end

inds = Vector{UnitRange}(undef, size(masks,2))
for i in 1:length(inds)
    inds[i] = maximum([mask[1] for mask in masks[:, i]]):minimum([mask[end] for mask in masks[:, i]])
end
# Something buggy if include order 1.  Order 86 essentially ignored due to tellurics. First and last orders have NaN issues
#max_orders = 12:83
lfc_orders = 43:72
order_list_timeseries = extract_orders(all_spectra,pipeline_plan; orders_to_use=lfc_orders, recalc=true)

times_nu = pipeline_plan.cache[:extract_orders].times
airmasses = [parse(Float64, md[:airmass]) for md in pipeline_plan.cache[:extract_orders].metadata]

## Switching to my data format

using JLD2
import telfitting; tf = telfitting

obs_resolution = 150000
desired_order = 50

n_obs = length(all_spectra)
extra_chop = 0
mask_inds = (inds[desired_order][1] + extra_chop):(inds[desired_order][end] - extra_chop)

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
tel_model_res = 2 * sqrt(2) * obs_resolution

@time tf_model = tf.TFOrderModel(tf_data, star_model_res, tel_model_res)

@time rvs_notel, rvs_naive = tf.initialize!(tf_model, tf_data; use_gp=true)

@save expres_data_path * star * ".jld2" tf_model n_obs tf_data rvs_naive rvs_notel times_nu airmasses
