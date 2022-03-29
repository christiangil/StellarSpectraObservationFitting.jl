## Importing packages
using Pkg
Pkg.activate("NEID")

import StellarSpectraObservationFitting as SSOF
using JLD2
using Statistics
import StatsBase

## Setting up necessary variables

stars = ["10700"]
star = stars[SSOF.parse_args(1, Int, 1)]
interactive = length(ARGS) == 0
save_plots = true
include("data_locs.jl")  # defines neid_data_path and neid_save_path
desired_order = SSOF.parse_args(2, Int, 81)  # 81 has a bunch of tels, 60 has very few
use_reg = SSOF.parse_args(3, Bool, true)
which_opt = SSOF.parse_args(4, Int, 3)
recalc = SSOF.parse_args(5, Bool, false)
oversamp = SSOF.parse_args(6, Bool, true)
use_lsf = SSOF.parse_args(7, Bool, false)
max_components = 5

## Loading in data and initializing model
save_path = neid_save_path * star * "/$(desired_order)/"
@load save_path * "data.jld2" n_obs data times_nu airmasses
if !use_reg
    save_path *= "noreg_"
end
if which_opt == 1
    save_path *= "optim_"
end
if !oversamp
    save_path *= "undersamp_"
end
if !use_lsf
    data = SSOF.GenericData(data)
    save_path *= "nolsf_"
end

# takes a couple mins now
if isfile(save_path*"results.jld2") && !recalc
    @load save_path*"results.jld2" model rvs_naive rvs_notel
    if model.metadata[:todo][:err_estimated]
        @load save_path*"results.jld2" rv_errors
    end
    if model.metadata[:todo][:downsized]
        @load save_path*"model_decision.jld2" comp_ls ℓ aics bics ks test_n_comp_tel test_n_comp_star
    end
else
    @time model = SSOF.OrderModel(data, "NEID", desired_order, star; n_comp_tel=max_components, n_comp_star=max_components, oversamp=oversamp)
    @time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(model, data)
    if !use_reg
        SSOF.rm_regularization(model)
        model.metadata[:todo][:reg_improved] = true
    end
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end

## Creating optimization workspace
if which_opt == 1
    mws = SSOF.OptimWorkspace(model, data)
elseif which_opt == 2
    mws = SSOF.TelStarWorkspace(model, data)
else
    mws = SSOF.TotalWorkspace(model, data)
end

## Plotting

SSOF_path = dirname(dirname(pathof(SSOF)))
if interactive
    include(SSOF_path * "/src/_plot_functions.jl")
    status_plot(mws)
else
    ENV["GKSwstype"] = "100"  # setting the GR workstation type to 100/nul
end

## Improving regularization

if !model.metadata[:todo][:reg_improved]  # 27 mins
    @time SSOF.train_OrderModel!(mws; print_stuff=true, ignore_regularization=true)  # 45s
    n_obs_test = Int(round(0.25 * n_obs))
    test_start_ind = max(1, Int(round(rand() * (n_obs - n_obs_test))))
    testing_inds = test_start_ind:test_start_ind+n_obs_test-1
    @time SSOF.fit_regularization!(mws, testing_inds)
    model.metadata[:todo][:reg_improved] = true
    model.metadata[:todo][:optimized] = false
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end

## Optimizing model

if !model.metadata[:todo][:optimized]
    @time results = SSOF.train_OrderModel!(mws; print_stuff=true)  # 120s
    rvs_notel_opt = SSOF.rvs(model)
    if interactive; status_plot(mws) end
    model.metadata[:todo][:optimized] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end

## Downsizing model

@time if !model.metadata[:todo][:downsized]  # 1.5 hrs (for 9x9)
    test_n_comp_tel = 0:max_components
    test_n_comp_star = 0:max_components
    ks = zeros(Int, length(test_n_comp_tel), length(test_n_comp_star))
    comp_ls = zeros(length(test_n_comp_tel), length(test_n_comp_star))
    comp_stds = zeros(length(test_n_comp_tel), length(test_n_comp_star))
    comp_intra_stds = zeros(length(test_n_comp_tel), length(test_n_comp_star))
    for (i, n_tel) in enumerate(test_n_comp_tel)
        for (j, n_star) in enumerate(test_n_comp_star)
            comp_ls[i, j], ks[i, j], comp_stds[i, j], comp_intra_stds[i, j] = SSOF.test_ℓ_for_n_comps([n_tel, n_star], mws, times_nu)
        end
    end
    n_comps_best, ℓ, aics, bics = SSOF.choose_n_comps(comp_ls, ks, test_n_comp_tel, test_n_comp_star, data.var; return_inters=true)
    @save save_path*"model_decision.jld2" comp_ls ℓ aics bics ks test_n_comp_tel test_n_comp_star comp_stds comp_intra_stds

    model_large = copy(model)
    model = SSOF.downsize(model, n_comps_best[1], n_comps_best[2])
    # model = SSOF.downsize(model, 1, 0)
    model.metadata[:todo][:downsized] = true
    model.metadata[:todo][:reg_improved] = true
    mws = typeof(mws)(model, data)
    SSOF.train_OrderModel!(mws; print_stuff=true)  # 120s
    SSOF.finalize_scores!(mws)
    model.metadata[:todo][:optimized] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel model_large
end

if save_plots
    include(SSOF_path * "/src/_plot_functions.jl")
    diagnostics = [ℓ, aics, bics, comp_stds, comp_intra_stds]
    diagnostics_labels = ["ℓ", "AIC", "BIC", "RV std", "Intra-night RV std"]
    diagnostics_fn = ["l", "aic", "bic", "rv", "rv_intra"]
    for i in 1:length(diagnostics)
        plt = component_test_plot(diagnostics[i], test_n_comp_tel, test_n_comp_star, ylabel=diagnostics_labels[i]);
        png(plt, save_path * diagnostics_fn[i] * "_choice.png")
    end
end

## Getting RV error bars (only regularization held constant)

@time if !model.metadata[:todo][:err_estimated] # 25 mins
    data.var[data.var.==Inf] .= 0
    data_noise = sqrt.(data.var)
    data.var[data.var.==0] .= Inf

    n = 50
    rv_holder = Array{Float64}(undef, n, length(model.rv.lm.s))
    _mws = typeof(mws)(copy(model), copy(data))
    _mws_score_finalizer() = SSOF.finalize_scores_setup(_mws)
    @time for i in 1:n
        _mws.d.flux .= data.flux .+ (data_noise .* randn(size(data.var)))
        SSOF.train_OrderModel!(_mws; iter=50)
        _mws_score_finalizer()
        rv_holder[i, :] = SSOF.rvs(_mws.om)
    end
    rv_errors = vec(std(rv_holder; dims=1))
    model.metadata[:todo][:err_estimated] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel rv_errors
end

## Plots

if save_plots
    include(SSOF_path * "/src/_plot_functions.jl")
    @load neid_save_path * star * "/neid_pipeline.jld2" neid_time neid_rv neid_rv_σ neid_order_rv ord_has_rvs

    # Compare RV differences to actual RVs from activity
    rvs_notel_opt = SSOF.rvs(model)
    plt = plot_model_rvs(times_nu, rvs_notel_opt, vec(rv_errors), neid_time, neid_rv, neid_rv_σ; display_plt=interactive, markerstrokewidth=1, title="HD$star (median σ: $(round(median(vec(rv_errors)), digits=3)))");
    png(plt, save_path * "model_rvs.png")

    if ord_has_rvs[desired_order]
        plt = plot_model_rvs(times_nu, rvs_notel_opt, vec(rv_errors), neid_time, neid_order_rv[:, desired_order], zeros(n_obs); display_plt=interactive, markerstrokewidth=1, title="HD$star (median σ: $(round(median(vec(rv_errors)), digits=3)))");
        png(plt, save_path * "model_rvs_order.png")
    end

    plt = plot_model(mws, airmasses; display_plt=interactive);
    png(plt, save_path * "model.png")

    plt = status_plot(mws; display_plt=interactive);
    png(plt, save_path * "status_plot.png")
end
