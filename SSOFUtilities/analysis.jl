## Some helpful analysis functions
import StellarSpectraObservationFitting as SSOF
using JLD2
using Statistics
import StatsBase
using Base.Threads
using ThreadsX
# using AdvancedHMC

valid_optimizers = ["adam", "l-bfgs", "frozen-tel"]


function get_data(data_fn::String; min_pix::Int=800, use_lsf::Bool=true)
	# save_path = save_path_base * star * "/$(desired_order)/"
	@load data_fn n_obs data times_nu airmasses
	if sum(all(.!(isinf.(data.var)); dims=2)) < min_pix
		@error "quitting analysis as there is not enough useful data (<$min_pix pixels used at all times)"
	end
	data.var[data.var .== 0] .= Inf
	if !use_lsf; data = SSOF.GenericData(data) end
	return data, times_nu, airmasses
end
function calculate_initial_model(data::SSOF.Data, instrument::String, desired_order::Int, star::String, times::AbstractVector;
	n_comp_tel::Int=5, n_comp_star::Int=5, save_fn::String="", plots_fn::String="",
	recalc::Bool=false, use_custom_n_comp::Bool=false, use_reg::Bool=true, kwargs...)

	save = save_fn!=""
	if isfile(save_fn) && !recalc
		println("using saved model at $save_fn")
		@load save_fn model
	else
		oms, ℓs, aics, bics, rv_stds, rv_stds_intra, _ = SSOF.calculate_initial_model(data, instrument, desired_order, star, times;
			max_n_tel=n_comp_tel, max_n_star=n_comp_star, use_all_comps=use_custom_n_comp, return_full_path=true, kwargs...)

		# plots
		if plots_fn!=""
			test_n_comp_tel = -1:n_comp_tel
			test_n_comp_star = 0:n_comp_star
			metrics = [ℓs, aics, bics, rv_stds, rv_stds_intra]
			metrics_labels = ["ℓ", "AIC", "BIC", "RV std", "Intra-night RV std"]
			metrics_fn = ["l", "aic", "bic", "rv", "rv_intra"]

			for i in eachindex(metrics)
				if any(isfinite.(metrics[i]))
					plt = component_test_plot(metrics[i], test_n_comp_tel, test_n_comp_star, ylabel=metrics_labels[i]);
					png(plt, plots_fn * metrics_fn[i] * "_choice.png")
				end
			end
		end

		model = oms[argmin(aics)]

		if !use_reg
			SSOF.rm_regularization!(model)
			model.metadata[:todo][:reg_improved] = true
		end
		if save
			@save save_fn model
		end
	end
	return model
end

function create_model(
	data_fn::String,
	desired_order::Int,
	instrument::String,
	star::String;
	n_comp_tel::Int=5,
	n_comp_star::Int=5,
	use_reg::Bool=true,
	save_fn::String="",
	recalc::Bool=false,
	min_pix::Int=800,
	kwargs...
	)

	save = save_fn!=""

	# save_path = save_path_base * star * "/$(desired_order)/"
	@load data_fn n_obs data times_nu airmasses
	if sum(all(.!(isinf.(data.var)); dims=2)) < min_pix
		@error "quitting analysis as there is not enough useful data (<$min_pix pixels used at all times)"
	end
	data.var[data.var .== 0] .= Inf

	if isfile(save_fn) && !recalc
		println("using saved model at $save_fn")
	    @load save_fn model
	else
	    model = SSOF.OrderModel(data, instrument, desired_order, star; n_comp_tel=n_comp_tel, n_comp_star=n_comp_star, kwargs...)
		if !use_reg
			SSOF.rm_regularization!(model)
			model.metadata[:todo][:reg_improved] = true
		end
		if save
			@save save_fn model
		end
	end
	return model, data, times_nu, airmasses
end
function initialize_model!(
	model::SSOF.OrderModel,
	data::SSOF.Data;
	init_fn::String="",
	recalc::Bool=false,
	kwargs...
	)

	save = init_fn!=""

	if isfile(init_fn) && !recalc
		@load init_fn lm_tel lm_star stellar_dominated
		if !model.metadata[:todo][:initialized]
			SSOF.fill_TelModel!(model, lm_tel[1])
			SSOF.fill_StarModel!(model, lm_star[1])
			model.metadata[:todo][:initialized] = true
		end
	else
		lm_tel, lm_star, stellar_dominated = SSOF.initializations!(model, data; kwargs...)
		# lm_tel, lm_star = SSOF.FullLinearModel[], SSOF.FullLinearModel[]
		if save
			@save init_fn lm_tel lm_star stellar_dominated
		end
	end
	return lm_tel, lm_star, stellar_dominated
end

function create_workspace(model, data, opt::String)
	@assert opt in valid_optimizers
	if opt == "l-bfgs"
		mws = SSOF.OptimTelStarWorkspace(model, data)
	elseif opt == "frozen-tel"
		mws = SSOF.FrozenTelWorkspace(model, data)
	else
		mws = SSOF.TotalWorkspace(model, data)
	end
	return mws
end

function improve_regularization!(mws::SSOF.ModelWorkspace; redo::Bool=false, verbose::Bool=true, testing_ratio::Real=0.33, save_fn::String="", kwargs...)

	save = save_fn!=""

	model = mws.om
	if redo || !model.metadata[:todo][:reg_improved]  # 27 mins
		@assert 0 < testing_ratio < 1
		n_obs = size(mws.d.flux, 2)

	    SSOF.train_OrderModel!(mws; verbose=verbose, ignore_regularization=true)  # 45s
	    n_obs_test = Int(round(testing_ratio * n_obs))
	    test_start_ind = max(1, Int(round(rand() * (n_obs - n_obs_test))))
	    testing_inds = test_start_ind:test_start_ind+n_obs_test-1
	    SSOF.fit_regularization!(mws, testing_inds; kwargs...)
	    model.metadata[:todo][:reg_improved] = true
	    if save; @save save_fn model end
	end
end

function improve_model!(mws::SSOF.ModelWorkspace; verbose::Bool=true, show_plot::Bool=false, save_fn::String="", kwargs...)
	save = save_fn!=""
	model = mws.om
	SSOF.train_OrderModel!(mws; verbose=verbose, kwargs...)  # 120s
	results = SSOF.finalize_scores!(mws; verbose=verbose, kwargs...)
    if show_plot; status_plot(mws) end
    if save; @save save_fn model end
	return results
end
function improve_model!(mws::SSOF.ModelWorkspace, airmasses::AbstractVector, times::AbstractVector; show_plot::Bool=false, kwargs...)
	results = improve_model!(mws; show_plot=show_plot, kwargs...)
	if show_plot; plot_model(mws, airmasses, times) end
	return results
end

function downsize_model(mws::SSOF.ModelWorkspace, times::AbstractVector, lm_tel::Vector{<:SSOF.LinearModel}, lm_star::Vector{<:SSOF.LinearModel}; save_fn::String="", decision_fn::String="", verbose::Bool=true, plots_fn::String="", iter::Int=50, ignore_regularization::Bool=true, multithread::Bool=nthreads() > 3, kwargs...)
	model = mws.om

	if !model.metadata[:todo][:downsized]  # 1.5 hrs (for 9x9)
		save = save_fn!=""
		save_md = decision_fn!=""
		save_plots = plots_fn!=""

	    SSOF.is_time_variable(model.tel.lm) ?
			test_n_comp_tel = (-axes(model.tel.lm.M, 2)) :
			test_n_comp_tel = (-1:0)
		SSOF.is_time_variable(model.star.lm) ?
			test_n_comp_star = (0:size(model.star.lm.M, 2)) :
			test_n_comp_star = (0:0)
	    ks = zeros(Int, length(test_n_comp_tel), length(test_n_comp_star))
	    comp_ls = zeros(length(test_n_comp_tel), length(test_n_comp_star))
	    comp_stds = zeros(length(test_n_comp_tel), length(test_n_comp_star))
	    comp_intra_stds = zeros(length(test_n_comp_tel), length(test_n_comp_star))
		better_models = zeros(Int, length(test_n_comp_tel), length(test_n_comp_star))
		if multithread
			# @threads for p_i in collect(Iterators.product(eachindex(test_n_comp_tel), eachindex(test_n_comp_star)))
			# 	i, j = p_i
			# 	comp_ls[i, j], ks[i, j], comp_stds[i, j], comp_intra_stds[i, j], better_models[i, j] = SSOF.test_ℓ_for_n_comps([test_n_comp_tel[i], test_n_comp_star[j]], mws, times, lm_tel, lm_star; iter=iter, ignore_regularization=ignore_regularization)
		    # end
			# using ThreadsX  # tiny bit better performance
			ThreadsX.foreach(collect(Iterators.product(eachindex(test_n_comp_tel), eachindex(test_n_comp_star)))) do (i, j)
			   comp_ls[i, j], ks[i, j], comp_stds[i, j], comp_intra_stds[i, j], better_models[i, j] = SSOF.test_ℓ_for_n_comps([test_n_comp_tel[i], test_n_comp_star[j]], mws, times, lm_tel, lm_star; iter=iter, ignore_regularization=ignore_regularization)
			end
		else
			for (i, n_tel) in enumerate(test_n_comp_tel)
				for (j, n_star) in enumerate(test_n_comp_star)
					comp_ls[i, j], ks[i, j], comp_stds[i, j], comp_intra_stds[i, j], better_models[i, j] = SSOF.test_ℓ_for_n_comps([n_tel, n_star], mws, times, lm_tel, lm_star; iter=iter, ignore_regularization=ignore_regularization)
				end
			end
		end
	    n_comps_best, ℓ, aics, bics, best_ind = SSOF.choose_n_comps(comp_ls, ks, test_n_comp_tel, test_n_comp_star, mws.d.var; return_inters=true, kwargs...)
	    if save_md; @save decision_fn comp_ls ℓ aics bics best_ind ks test_n_comp_tel test_n_comp_star comp_stds comp_intra_stds better_models end

	    # model_large = copy(model)
		mws_smol = _downsize_model(mws, n_comps_best, better_models[best_ind], lm_tel, lm_star; verbose=verbose, iter=iter, ignore_regularization=ignore_regularization)

		if save_plots
			diagnostics = [ℓ, aics, bics, comp_stds, comp_intra_stds]
			diagnostics_labels = ["ℓ", "AIC", "BIC", "RV std", "Intra-night RV std"]
			diagnostics_fn = ["l", "aic", "bic", "rv", "rv_intra"]
			for i in eachindex(diagnostics)
				if !all(isinf.(diagnostics[i]))
					plt = component_test_plot(diagnostics[i], test_n_comp_tel, test_n_comp_star, ylabel=diagnostics_labels[i]);
					png(plt, plots_fn * diagnostics_fn[i] * "_choice.png")
				end
			end
		end

		if save; @save save_fn model end

		return mws_smol#, ℓ, aics, bics, comp_stds, comp_intra_stds
	end
	return mws
end
function _finish_downsizing(mws::SSOF.ModelWorkspace, model::SSOF.OrderModel; no_tels::Bool=false, kwargs...)
	if no_tels
		mws_smol = SSOF.FrozenTelWorkspace(model, mws.d)
		model.tel.lm.μ .= 1
	else
		mws_smol = typeof(mws)(model, mws.d)
	end
	mws_smol.om.metadata[:todo][:downsized] = true
	mws_smol.om.metadata[:todo][:initialized] = true
	SSOF.flip_basis_vectors!(mws_smol.om)
	SSOF.train_OrderModel!(mws_smol; kwargs...)  # 120s
	SSOF.finalize_scores!(mws_smol; f_tol=SSOF._f_reltol_def_s, g_tol=SSOF._g_L∞tol_def_s)
	return mws_smol
end
function _downsize_model(mws::SSOF.ModelWorkspace, n_comps::Vector{<:Int}, better_model::Int, lm_tel::Vector{<:SSOF.LinearModel}, lm_star::Vector{<:SSOF.LinearModel}; lm_tel_ind::Int=n_comps[2]+1, lm_star_ind::Int=n_comps[1]+1, kwargs...)
	model = SSOF.downsize(mws.om, max(0, n_comps[1]), n_comps[2])
	model.metadata[:todo][:downsized] = true
	if all(n_comps .> 0)
		SSOF._fill_model!(model, n_comps, better_model, lm_tel, lm_star; lm_tel_ind=lm_tel_ind, lm_star_ind=lm_star_ind)
		better_model==1 ?
			println("downsizing: used the $(n_comps[1]) telluric basis -> $(n_comps[2]) stellar basis initialization") :
			println("downsizing: used the $(n_comps[2]) stellar basis -> $(n_comps[1]) telluric basis initialization")
	end
	return _finish_downsizing(mws, model; no_tels=n_comps[1]<0, kwargs...)
end
function _downsize_model(mws::SSOF.ModelWorkspace, n_comps::Vector{<:Int}; kwargs...)
	model = SSOF.downsize(mws.om, max(0, n_comps[1]), n_comps[2])
	return _finish_downsizing(mws, model; no_tels=n_comps[1]<0, kwargs...)
end


function estimate_σ_curvature(mws::SSOF.ModelWorkspace; recalc::Bool=false, save_fn::String="", kwargs...)
	save = save_fn!=""
	model = mws.om
	if recalc || !model.metadata[:todo][:err_estimated] # 25 mins
	    mws.d.var[mws.d.var.==Inf] .= 0
	    data_noise = sqrt.(mws.d.var)
	    mws.d.var[mws.d.var.==0] .= Inf

		time_var_tel = SSOF.is_time_variable(model.tel)
		time_var_star = SSOF.is_time_variable(model.star)

		typeof(model) <: SSOF.OrderModelDPCA ? rvs = copy(model.rv.lm.s) : rvs = copy(model.rv)
		ℓ_rv(x) = SSOF._loss(mws.o, model, mws.d; rv=x) / 2  # factor of 2 makes curvature estimates correct (χ² -> data fit part of multivariate Gaussian)
		rvs_σ = estimate_σ_curvature_helper(rvs, ℓ_rv; param_str="rv", kwargs...)
		if typeof(model) <: SSOF.OrderModelDPCA
			rvs = vec(rvs)
			rvs .*= -SSOF.light_speed_nu
			rvs_σ .*= SSOF.light_speed_nu
		end

		if time_var_tel
			ℓ_tel(x) = (SSOF._loss(mws.o, model, mws.d; tel=vec(model.tel.lm)) + SSOF.model_s_prior(model.tel.lm.s, model.reg_tel)) / 2  # factor of 2 makes curvature estimates correct (χ² -> data fit part of multivariate Gaussian)
			tel_s_σ = estimate_σ_curvature_helper(model.tel.lm.s, ℓ_tel; param_str="tel_s", kwargs...)
		else
			tel_s_σ = nothing
		end

		if time_var_star
			ℓ_star(x) = (SSOF._loss(mws.o, model, mws.d; star=vec(model.star.lm)) + SSOF.model_s_prior(model.star.lm.s, model.reg_star)) / 2  # factor of 2 makes curvature estimates correct (χ² -> data fit part of multivariate Gaussian)
			star_s_σ = estimate_σ_curvature_helper(model.star.lm.s, ℓ_star; param_str="star_s", kwargs...)
		else
			star_s_σ = nothing
		end

		model.metadata[:todo][:err_estimated] = true
	    if save; @save save_fn rvs rvs_σ tel_s_σ star_s_σ end
		return rvs, rvs_σ, tel_s_σ, star_s_σ
	else
		println("loading σs")
		if save; @load save_fn rvs rvs_σ tel_s_σ star_s_σ end
		return rvs, rvs_σ, tel_s_σ, star_s_σ
	end
end


function estimate_σ_curvature_helper(x::AbstractVecOrMat, ℓ::Function; n::Int=7, use_gradient::Bool=false, multithread::Bool=nthreads() > 3, print_every::Int=10, kwargs...)
	σs = Array{Float64}(undef, length(x))
	if !multithread
		x_test = Array{Float64}(undef, n)
		ℓs = Array{Float64}(undef, n)
	end
	if use_gradient; g = ∇(ℓ) end
	_std = std(x)
	if multithread
		nchains = nthreads()
		schedule = collect(Iterators.partition(eachindex(x), Int(ceil(length(x)/nchains))))
		# Threads.@threads for i in 1:nchains
		ThreadsX.foreach(1:nchains) do i
			local _todo = copy(schedule[i])
			local _σs = Array{Float64}(undef, length(_todo))
			local _x = copy(x)
			local _x_test = Array{Float64}(undef, n)
			local _ℓs = Array{Float64}(undef, n)
			for ii in eachindex(_todo)
				k = _todo[ii]
				_x_test .= _x[k] .+ LinRange(-_std, _std, n)
				for j in 1:n
					_x[k] = _x_test[j]
					if use_gradient
						_ℓs[j] = only(g(_x))[k]
					else
						_ℓs[j] = ℓ(_x)
					end
				end
				estimate_σ_curvature_helper_finalizer!(_σs, _ℓs, _x_test, ii; use_gradient=use_gradient, print_every=100000, kwargs...)
			end
			σs[_todo] .= _σs
		end
	else
		for i in eachindex(x)
			hold = x[i]
			# x_test[:] = x[i] .+ LinRange(-_std/1e3, _std/1e3, n)
			x_test[:] = x[i] .+ LinRange(-_std, _std, n)
			for j in 1:n
				x[i] = x_test[j]
				if use_gradient
					ℓs[j] = only(g(x))[i]
				else
					ℓs[j] = ℓ(x)
				end
			end
			x[i] = hold
			# println("$i: ", ℓs .- ℓs[Int(round(n//2))])
			estimate_σ_curvature_helper_finalizer!(σs, ℓs, x_test, i; use_gradient=use_gradient, print_every=print_every, kwargs...)
		end
	end
	return reshape(σs, size(x))
end

function estimate_σ_curvature_helper_finalizer!(σs::AbstractVecOrMat, _ℓs::AbstractVector, x_test::AbstractVector, i::Int; use_gradient::Bool=false, param_str::String="", print_every::Int=10, verbose::Bool=false, show_plots::Bool=false, )
	if use_gradient
		poly_f = SSOF.ordinary_lst_sq_f(_ℓs, 1; x=x_test)
		σs[i] = sqrt(1 / poly_f.w[2])
		max_dif = maximum(abs.((poly_f.(x_test)./_ℓs) .- 1))
		if verbose; println("∇_$i: $(poly_f.w[1] + poly_f.w[2] * x[i])") end
	else
		poly_f = SSOF.ordinary_lst_sq_f(_ℓs, 2; x=x_test)
		σs[i] = sqrt(1 / (2 * poly_f.w[3]))
		max_dif = maximum(abs.((poly_f.(x_test)./_ℓs) .- 1))
		if verbose; println("∇_$i: $(poly_f.w[2] + 2 * poly_f.w[3] * x[i])") end
	end
	if show_plots
		plt = scatter(x_test, _ℓs; label="ℓ")
		plot!(x_test, poly_f.(x_test); label="polynomial fit")
		display(plt)
	end
	if max_dif > 1e-2; @warn param_str * "_σ[$i] misfit at $(round(100*max_dif; digits=2))% level" end
	if i%print_every==0; println("done with $i/$(length(σs)) " * param_str * "_σ estimates") end
	# println("done with $i/$(length(σs)) " * param_str * "_σ estimates")
end

function estimate_σ_bootstrap_reducer(shaper::AbstractArray, holder::AbstractArray, reducer::Function)
	result = Array{Float64}(undef, size(shaper, 1), size(shaper, 2))
	for i in axes(shaper, 1)
		result[i, :] .= vec(reducer(view(holder, :, i, :); dims=1))
	end
	return result
end

function estimate_σ_bootstrap_helper!(rv_holder::AbstractMatrix, tel_holder, star_holder, i::Int, mws::SSOF.ModelWorkspace, data_noise::AbstractMatrix, n::Int; verbose::Bool=true)
	time_var_tel = SSOF.is_time_variable(mws.om.tel)
	time_var_star = SSOF.is_time_variable(mws.om.star)
	_mws = typeof(mws)(copy(mws.om), copy(mws.d))
	_mws.d.flux .= mws.d.flux .+ (data_noise .* randn(size(mws.d.var)))
	improve_model!(_mws, iter=50, verbose=false)
	rv_holder[i, :] = SSOF.rvs(_mws.om)
	if time_var_tel
		tel_holder[i, :, :] .= _mws.om.tel.lm.s
	end
	if time_var_star
		star_holder[i, :, :] .= _mws.om.star.lm.s
	end
	if (verbose && i%10==0); println("done with $i/$n bootstraps") end
end

function estimate_σ_bootstrap(mws::SSOF.ModelWorkspace; recalc::Bool=false, save_fn::String="", n::Int=50, return_holders::Bool=false, recalc_mean::Bool=false, multithread::Bool=nthreads() > 3, verbose::Bool=true)
	save = save_fn!=""
	if recalc || !mws.om.metadata[:todo][:err_estimated] # 25 mins
	    mws.d.var[mws.d.var.==Inf] .= 0
	    data_noise = sqrt.(mws.d.var)
	    mws.d.var[mws.d.var.==0] .= Inf

	    typeof(mws.om) <: SSOF.OrderModelWobble ?
		 	rv_holder = Array{Float64}(undef, n, length(mws.om.rv)) :
			rv_holder = Array{Float64}(undef, n, length(mws.om.rv.lm.s))

		time_var_tel = SSOF.is_time_variable(mws.om.tel)
		time_var_star = SSOF.is_time_variable(mws.om.star)
		time_var_tel ?
			tel_holder = Array{Float64}(undef, n, size(mws.om.tel.lm.s, 1), size(mws.om.tel.lm.s, 2)) :
			tel_holder = nothing
		time_var_star ?
			star_holder = Array{Float64}(undef, n, size(mws.om.star.lm.s, 1), size(mws.om.star.lm.s, 2)) :
			star_holder = nothing
		if multithread
			# @threads for i in 1:n
			# # using Polyester  # same performance
			# @batch per=core for i in 1:n
			# using ThreadsX  # tiny bit better performance
			ThreadsX.foreach(1:n) do i
				estimate_σ_bootstrap_helper!(rv_holder, tel_holder, star_holder, i, mws, data_noise, n; verbose=false)
			end
		else
			for i in 1:n
				estimate_σ_bootstrap_helper!(rv_holder, tel_holder, star_holder, i, mws, data_noise, n; verbose=verbose)
			end
	    end
		recalc_mean ? rvs = vec(mean(rv_holder; dims=1)) : rvs = SSOF.rvs(mws.om)
	    rvs_σ = vec(std(rv_holder; dims=1))

		if time_var_tel
			recalc_mean ?
				tel_s = estimate_σ_bootstrap_reducer(mws.om.tel.lm.s, tel_holder, mean) :
				tel_s = mws.om.tel.lm.s
			tel_s_σ = estimate_σ_bootstrap_reducer(mws.om.tel.lm.s, tel_holder, std)
		else
			tel_s = nothing
			tel_s_σ = nothing
		end
		if time_var_star
			recalc_mean ?
				star_s = estimate_σ_bootstrap_reducer(mws.om.star.lm.s, star_holder, mean) :
				star_s = mws.om.star.lm.s
			star_s_σ = estimate_σ_bootstrap_reducer(mws.om.star.lm.s, star_holder, std)
		else
			star_s = nothing
			star_s_σ = nothing
		end
		mws.om.metadata[:todo][:err_estimated] = true
	    if save; @save save_fn rvs rvs_σ tel_s tel_s_σ star_s star_s_σ end
		if return_holders
			return rvs, rvs_σ, tel_s, tel_s_σ, star_s, star_s_σ, rv_holder, tel_holder, star_holder
		else
			return rvs, rvs_σ, tel_s, tel_s_σ, star_s, star_s_σ
		end
	else
		@assert isfile(save_fn)
		println("loading rvs")
		if save; @load save_fn rvs rvs_σ tel_s tel_s_σ star_s star_s_σ end
		return rvs, rvs_σ, tel_s, tel_s_σ, star_s, star_s_σ
	end
end


# # the code is slow and doesn't seem to work
# function estimate_σ_mcmc(mws::SSOF.ModelWorkspace; recalc::Bool=false, save_fn::String="", n::Int=52, return_holders::Bool=false, recalc_mean::Bool=false, multithread::Bool=nthreads() > 3, verbose::Bool=true)
# 	save = save_fn!=""
# 	if recalc || !mws.om.metadata[:todo][:err_estimated] # 25 mins
#
# 		@assert typeof(mws.om) <: SSOF.TotalWorkspace  # TODO add OptimWorkspace verison
# 		# initial_θ, obj, unflatten = SSOF.opt_funcs(mws.total.l, mws.total.θ)
# 		mws_s = typeof(mws)(mws.om, mws.d; only_s=true)
# 		initial_θ, obj, unflatten, g_nabla, g_val_nabla = SSOF.opt_funcs(mws_s.total.l, mws_s.total.θ)
# 		D = length(initial_θ)
#
# 		ℓπ(θ) = -obj.f(θ)
# 		function dℓπ(θ)
# 			nl, ng_unfl = g_val_nabla(unflatten(θ))
# 			ng, _ = SSOF.flatten(only(ng_unfl))
# 			return -nl.val, -ng
# 		end
#
# 		n_adapts = Int(round(n / 5))
#
# 		# Define a Hamiltonian system
# 		metric = DiagEuclideanMetric(D)
# 		hamiltonian = Hamiltonian(metric, ℓπ, dℓπ)
#
# 		# Define a leapfrog solver, with initial step size chosen heuristically
# 		initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
# 		integrator = Leapfrog(initial_ϵ)
#
# 		# Define an HMC sampler, with the following components
# 		#   - multinomial sampling scheme,
# 		#   - generalised No-U-Turn criteria, and
# 		#   - windowed adaption for step-size and diagonal mass matrix
# 		proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
# 		adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
# 		if multithread
# 			# Number of chains to sample
# 			nchains = nthreads()
# 			chains = Vector{Any}(undef, nchains)
# 			n_per_chain = Int(round(n/nchains))
# 			# Threads.@threads for i in 1:nchains
# 			ThreadsX.foreach(1:nchains) do i
# 				_samples, _ = sample(hamiltonian, proposal, initial_θ, n_per_chain, adaptor, n_adapts; verbose=false)
# 				chains[i] = _samples
# 			end
# 			# samples = reduce(hcat,Iterators.flatten(chains))
# 			samples = Iterators.flatten(chains)
# 		else
# 			# Run the sampler, where
# 			#   - `samples` will store the samples
# 			#   - `stats` will store diagnostic statistics for each sample
# 			samples, _ = sample(hamiltonian, proposal, initial_θ, n, adaptor, n_adapts; progress=true)
# 			# samples = reduce(hcat,samples)
# 		end
# 		samples = reduce(hcat, samples)
# 		samples = [unflatten(view(samples, :, i)) for i in axes(samples, 2)]
# 		typeof(mws.om) <: SSOF.OrderModelWobble ?
# 			rv_holder = Array{Float64}(undef, n, length(mws.om.rv)) :
# 			rv_holder = Array{Float64}(undef, n, length(mws.om.rv.lm.s))
#
# 		time_var_tel = SSOF.is_time_variable(mws.om.tel)
# 		time_var_star = SSOF.is_time_variable(mws.om.star)
# 		time_var_tel ?
# 			tel_holder = Array{Float64}(undef, n, size(mws.om.tel.lm.s, 1), size(mws.om.tel.lm.s, 2)) :
# 			tel_holder = nothing
# 		time_var_star ?
# 			star_holder = Array{Float64}(undef, n, size(mws.om.star.lm.s, 1), size(mws.om.star.lm.s, 2)) :
# 			star_holder = nothing
#
# 		for i in eachindex(samples)
# 			rv_holder[i, :] .= samples[i][end]
# 		end
# 		if time_var_tel
# 			tel_holder[i, :, :] .= samples[i][1]
# 		end
# 		if time_var_star
# 			star_holder[i, :, :] .= samples[i][time_var_tel+1]
# 		end
#
# 		recalc_mean ? rvs = vec(mean(rv_holder; dims=1)) : rvs = SSOF.rvs(mws.om)
# 	    rvs_σ = vec(std(rv_holder; dims=1))
#
# 		if time_var_tel
# 			recalc_mean ?
# 				tel_s = estimate_σ_bootstrap_reducer(mws.om.tel.lm.s, tel_holder, mean) :
# 				tel_s = mws.om.tel.lm.s
# 			tel_s_σ = estimate_σ_bootstrap_reducer(mws.om.tel.lm.s, tel_holder, std)
# 		else
# 			tel_s = nothing
# 			tel_s_σ = nothing
# 		end
# 		if time_var_star
# 			recalc_mean ?
# 				star_s = estimate_σ_bootstrap_reducer(mws.om.star.lm.s, star_holder, mean) :
# 				star_s = mws.om.star.lm.s
# 			star_s_σ = estimate_σ_bootstrap_reducer(mws.om.star.lm.s, star_holder, std)
# 		else
# 			star_s = nothing
# 			star_s_σ = nothing
# 		end
# 		mws.om.metadata[:todo][:err_estimated] = true
# 	    if save; @save save_fn rvs rvs_σ tel_s tel_s_σ star_s star_s_σ end
# 		if return_holders
# 			return rvs, rvs_σ, tel_s, tel_s_σ, star_s, star_s_σ, rv_holder, tel_holder, star_holder
# 		else
# 			return rvs, rvs_σ, tel_s, tel_s_σ, star_s, star_s_σ
# 		end
# 	else
# 		@assert isfile(save_fn)
# 		println("loading rvs")
# 		if save; @load save_fn rvs rvs_σ tel_s tel_s_σ star_s star_s_σ end
# 		return rvs, rvs_σ, tel_s, tel_s_σ, star_s, star_s_σ
# 	end
# end

# # the example code is slow as heck so not going to try to implement
# function estimate_σ_vi(mws::SSOF.ModelWorkspace; recalc::Bool=false, save_fn::String="", n::Int=52, return_holders::Bool=false, recalc_mean::Bool=false, multithread::Bool=nthreads() > 3, verbose::Bool=true)
# 	Pkg.add("DistributionsAD")
# 	Pkg.add("AdvancedVI")
# 	using DistributionsAD, AdvancedVI
# 	using Distributions
#
# 	d = 2; n = 100;
#
# 	observations = randn((d, n)); # 100 observations from 2D 𝒩(0, 1)
#
# 	prior(μ) = logpdf(MvNormal(ones(d)), μ)
# 	likelihood(x, μ) = sum(logpdf(MvNormal(μ, ones(d)), x))
# 	logπ(μ) = likelihood(observations, μ) + prior(μ)
# 	getq(θ) = TuringDiagMvNormal(θ[1:d], exp.(θ[d + 1:4]))
# 	advi = ADVI(10, 10_000)
# 	q = vi(logπ, advi, getq, randn(4))
# end


function how_many_comps(str::String, recalc::Bool, desired_order::Int)

	# they said to look for a file
	if isfile(str)
		df_n_comp = DataFrame(CSV.File(str))
		i_df = SSOF.int2ind(df_n_comp.order, desired_order)
		@assert df_n_comp[i_df, :order] == desired_order
		function look(field, default)
			try
				return df_n_comp[i_df, field]
			catch err
				if typeof(err) <: ArgumentError
					return default
				end
			end
		end
		use_custom_n_comp = look(:redo, true)
		recalc = recalc || use_custom_n_comp
		n_comp_tel = look(:n_tel_by_eye, 5)
		n_comp_star = look(:n_star_by_eye, 5)
		# better_model = look(:better_model, 1)
		# remove_reciprocal_continuum = look(:has_reciprocal_continuum, false)
		println("using $n_comp_tel telluric components and $n_comp_star stellar components from " * str)

	# they passed a proposed amount of parameters
elseif length(str) > 0 && str[1] == '[' && str[end] == ']'
		matches = [parse(Int64, t.match) for t in eachmatch(r"-?[0-9]+", str)]
		@assert length(matches) == 2 "should only pass things of the form [n_comps_tel::Int, n_comps_star::Int]"
		n_comp_tel = matches[1]
		n_comp_star = matches[2]
		# length(matches) < 3 ? better_model = 1 : better_model = matches[3]
		# @assert 0 < better_model < 3
		use_custom_n_comp = true
		recalc = true
		println("using $n_comp_tel telluric components and $n_comp_star stellar components")

	# using the defaults
	else
		n_comp_tel = 5
		n_comp_star = 5
		# better_model = 1  # this will be ignored later
		use_custom_n_comp = false
		println("using $n_comp_tel telluric components and $n_comp_star stellar components (the default)")
	end

	return n_comp_tel, n_comp_star, use_custom_n_comp, recalc
end
