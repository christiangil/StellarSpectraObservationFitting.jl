using TemporalGPs, Stheno, Distributions

struct TFData{T<:Real}
    flux::AbstractMatrix{T}
    var::AbstractMatrix{T}
    log_λ_obs::AbstractMatrix{T}
    log_λ_star::AbstractMatrix{T}
	function TFData(flux::AbstractMatrix{T}, var, log_λ_obs, log_λ_star) where {T<:Real}
		@assert size(flux) == size(var) == size(log_λ_obs) == size(log_λ_star)
		return new{T}(flux, var, log_λ_obs, log_λ_star)
	end
end
(tfd::TFData)(inds::AbstractVecOrMat) =
	TFData(view(tfd.flux, :, inds), view(tfd.var, :, inds),
	view(tfd.log_λ_obs, :, inds), view(tfd.log_λ_star, :, inds))


function create_λ_template(log_λ_obs, resolution)
    log_min_wav, log_max_wav = [minimum(log_λ_obs), maximum(log_λ_obs)]
    len = Int(ceil((exp(log_max_wav) - exp(log_min_wav)) * resolution / exp((log_max_wav + log_min_wav)/2)))
    log_Δλ = (log_max_wav - log_min_wav) / len
    len += 2
    log_λ_template = range(log_min_wav - log_Δλ; length = len,  stop = log_max_wav + log_Δλ)
    λ_template = exp.(log_λ_template)
    return len, log_λ_template, λ_template
end

struct LinearInterpolationHelper{T<:Number}
    li::AbstractMatrix{Int}  # lower indices
    ratios::AbstractMatrix{T}
	function LinearInterpolationHelper(
		li::AbstractMatrix{Int},  # lower indices
	    ratios::AbstractMatrix{T}) where {T<:Number}
		@assert size(li) == size(ratios)
		@assert all(0 .<= ratios .<= 1)
		# @assert some issorted thing?
		return new{T}(li, ratios)
	end
end
function (lih::LinearInterpolationHelper)(inds::AbstractVecOrMat, n_in::Int)
	n_out, n_obs = size(lih.li)
	@assert all(0 .< inds .<= n_obs)
	@assert allunique(inds)
	difference = 0
	j = inds[1]-1
	new_li = ones(Int, n_out, length(inds))
	new_li[:, 1] = lih.li[:, inds[1]] .- (j * n_in)
	for i in 2:length(inds)
		j += (inds[i] - inds[i - 1]) - 1
		new_li[:, i] = lih.li[:, inds[i]] .- (j * n_in)
	end
	return LinearInterpolationHelper(new_li, view(lih.ratios, :, inds))
end


function LinearInterpolationHelper_maker(to_λs::AbstractVecOrMat, from_λs::AbstractMatrix)
	len_from, n_obs = size(from_λs)
	len_to = size(to_λs, 1)

	lower_inds_t2f = zeros(Int, (len_from, n_obs))
	lower_inds_f2t = zeros(Int, (len_to, n_obs))
	ratios_t2f = zeros((len_from, n_obs))
	ratios_f2t = zeros((len_to, n_obs))

	function helper!(j, len, lower_inds, ratios, λs1, λs2)
		lower_inds[:, j] = searchsortednearest(λs1, λs2; lower=true)
		for i in 1:size(lower_inds, 1)
			# if point is after the end, just say the point is at the end
			if lower_inds[i, j] >= len
				lower_inds[i, j] = len - 1
				ratios[i, j] = 1
			# if point is before the start, keep ratios to be 0
			elseif λs2[i] >= λs1[lower_inds[i, j]]
				x0 = λs1[lower_inds[i, j]]
				x1 = λs1[lower_inds[i, j]+1]
				ratios[i, j] = (λs2[i] - x0) / (x1 - x0)
			end
			@assert 0 <= ratios[i,j] <= 1 "something is off with ratios[$i,$j] = $(ratios[i,j])"
		end
	end

	for j in 1:n_obs
    	current_from_λs = view(from_λs, :, j)
		helper!(j, len_to, lower_inds_t2f, ratios_t2f, to_λs, current_from_λs)
		helper!(j, len_from, lower_inds_f2t, ratios_f2t, current_from_λs, to_λs)
		lower_inds_t2f[:, j] .+= (j - 1) * len_to
		lower_inds_f2t[:, j] .+= (j - 1) * len_from
	end

	return LinearInterpolationHelper(lower_inds_t2f, ratios_t2f), LinearInterpolationHelper(lower_inds_f2t, ratios_f2t)
end


abstract type LinearModel end

# Full (includes mean) linear model
struct FullLinearModel{T<:Number} <: LinearModel
	M::AbstractMatrix{T}
	s::AbstractMatrix{T}
	μ::Vector{T}
	function FullLinearModel(M::AbstractMatrix{T}, s, μ) where {T<:Number}
		@assert length(μ) == size(M, 1)
		@assert size(M, 2) == size(s, 1)
		return new{T}(M, s, μ)
	end
end
# Base (no mean) linear model
struct BaseLinearModel{T<:Number} <: LinearModel
	M::AbstractMatrix{T}
	s::AbstractMatrix{T}
	function BaseLinearModel(M::AbstractMatrix{T}, s) where {T<:Number}
		@assert size(M, 2) == size(s, 1)
		return new{T}(M, s)
	end
end
_eval_lm(M, s) = M * s
_eval_full_lm(M, s, μ) = _eval_lm(M, s) .+ μ
(flm::FullLinearModel)() = _eval_full_lm(flm.M, flm.s, flm.μ)
(blm::BaseLinearModel)() = _eval_lm(blm.M, blm.s)
(flm::FullLinearModel)(inds::AbstractVecOrMat) = FullLinearModel(flm.M, view(flm.s, :, inds), flm.μ)
(blm::BaseLinearModel)(inds::AbstractVecOrMat) = BaseLinearModel(blm.M, view(blm.s, :, inds))

struct TFSubmodel{T<:Number}
    log_λ::AbstractVector{T}
    λ::AbstractVector{T}
	lm::LinearModel
    function TFSubmodel(log_λ_obs::AbstractVecOrMat, model_res::Real, n_comp::Int; include_mean::Bool=true)
        n_obs = size(log_λ_obs, 2)
		len, log_λ, λ = create_λ_template(log_λ_obs, model_res)
		if include_mean
			lm = FullLinearModel(zeros(len, n_comp), zeros(n_comp, n_obs), ones(len))
		else
			lm = BaseLinearModel(zeros(len, n_comp), zeros(n_comp, n_obs))
		end
        return TFSubmodel(log_λ, λ, lm)
    end
    function TFSubmodel(log_λ::AbstractVector{T}, λ, lm) where {T<:Number}
        @assert length(log_λ) == length(λ) == size(lm.M, 1)
        return new{T}(log_λ, λ, lm)
    end
end
(tfsm::TFSubmodel)(inds::AbstractVecOrMat) =
	TFSubmodel(tfsm.log_λ, tfsm.λ, tfsm.lm(inds))

function _shift_log_λ_model(log_λ_obs_from, log_λ_obs_to, log_λ_model_from)
	n_obs = size(log_λ_obs_from, 2)
	dop = [log_λ_obs_from[1, i] - log_λ_obs_to[1, i] for i in 1:n_obs]
	log_λ_model_to = ones(length(log_λ_model_from), n_obs)
	for i in 1:n_obs
		log_λ_model_to[:, i] = log_λ_model_from .+ dop[i]
	end
	return log_λ_model_to
end

struct TFOrderModel{T<:Number}
    tel::TFSubmodel{T}
    star::TFSubmodel{T}
	rv::TFSubmodel{T}
	reg_tel::Dict{Symbol, T}
	reg_star::Dict{Symbol, T}
    lih_t2b::LinearInterpolationHelper
    lih_b2t::LinearInterpolationHelper
    lih_o2b::LinearInterpolationHelper
    lih_b2o::LinearInterpolationHelper
    lih_t2o::LinearInterpolationHelper
    lih_o2t::LinearInterpolationHelper
	todo::Dict{Symbol, Bool}
	instrument::String
	order::Int
    function TFOrderModel(
		tfd::TFData,
		star_model_res::Number,
		tel_model_res::Number,
		instrument::String,
		order::Int;
		n_comp_tel::Int=2,
		n_comp_star::Int=2)

        tel = TFSubmodel(tfd.log_λ_obs, tel_model_res, n_comp_tel)
        star = TFSubmodel(tfd.log_λ_star, star_model_res, n_comp_star)
		rv = TFSubmodel(tfd.log_λ_star, star_model_res, 1; include_mean=false)

        n_obs = size(tfd.log_λ_obs, 2)
        lih_t2o, lih_o2t = LinearInterpolationHelper_maker(tel.log_λ, tfd.log_λ_obs)
        lih_b2o, lih_o2b = LinearInterpolationHelper_maker(star.log_λ, tfd.log_λ_star)
        star_dop = [tfd.log_λ_star[1, i] - tfd.log_λ_obs[1, i] for i in 1:n_obs]
        star_log_λ_tel = ones(length(star.log_λ), n_obs)
        for i in 1:n_obs
            star_log_λ_tel[:, i] = star.log_λ .+ star_dop[i]
        end
        lih_t2b, lih_b2t = LinearInterpolationHelper_maker(tel.log_λ, star_log_λ_tel)
		# reg_tel = Dict([(:shared_M, 1e-4), (:L2_μ, 1e4), (:L1_μ, 1e3),
		# 	(:L1_μ₊_factor, 2), (:L2_M, 1e7), (:L1_M, 1e4)])
		# reg_star = Dict([(:shared_M, 1e-4), (:L2_μ, 1e3), (:L1_μ, 1e-1),
		# 	(:L1_μ₊_factor, 2), (:L2_M, 1e7), (:L1_M, 1e6)])
		reg_tel = Dict([(:L2_μ, 1e4), (:L1_μ, 1e3),
			(:L1_μ₊_factor, 2), (:L2_M, 1e7), (:L1_M, 1e4)])
		reg_star = Dict([(:L2_μ, 1e3), (:L1_μ, 1e-1),
			(:L1_μ₊_factor, 2), (:L2_M, 1e7), (:L1_M, 1e6)])
		todo = Dict([(:reg_improved, false), (:extra_chop, false)])
        return TFOrderModel(tel, star, rv, reg_tel, reg_star, lih_t2b, lih_b2t,
			lih_o2b, lih_b2o, lih_t2o, lih_o2t, todo, instrument, order)
    end
    function TFOrderModel(tel::TFSubmodel{T}, star, rv, reg_tel, reg_star, lih_t2b,
		lih_b2t, lih_o2b, lih_b2o, lih_t2o, lih_o2t, todo, instrument, order) where {T<:Number}
		return new{T}(tel, star, rv, reg_tel, reg_star, lih_t2b, lih_b2t, lih_o2b, lih_b2o, lih_t2o, lih_o2t, todo, instrument, order)
	end
end
(tfom::TFOrderModel)(inds::AbstractVecOrMat) =
	TFOrderModel(tfom.tel(inds), tfom.star(inds), tfom.rv(inds), tfom.reg_tel, tfom.reg_star,
	tfom.lih_t2b(inds, size(tfom.lih_o2t.li, 1)), tfom.lih_b2t(inds, size(tfom.lih_o2b.li, 1)),
	tfom.lih_o2b(inds, size(tfom.lih_b2o.li, 1)), tfom.lih_b2o(inds, size(tfom.lih_o2b.li, 1)),
	tfom.lih_t2o(inds, size(tfom.lih_o2t.li, 1)), tfom.lih_o2t(inds, size(tfom.lih_b2o.li, 1)),
	tfom.todo, tfom.instrument, tfom.order)

downsize(lm::FullLinearModel, n_comp::Int) =
	FullLinearModel(lm.M[:, 1:n_comp], lm.s[1:n_comp, :], lm.μ[:])
downsize(lm::BaseLinearModel, n_comp::Int) =
	BaseLinearModel(lm.M[:, 1:n_comp], lm.s[1:n_comp, :])
downsize(tfsm::TFSubmodel, n_comp::Int) =
	TFSubmodel(tfsm.log_λ[:], tfsm.λ[:], downsize(tfsm.lm, n_comp))
downsize(tfm::TFOrderModel, n_comp_tel::Int, n_comp_star::Int) =
	TFOrderModel(
		downsize(tfm.tel, n_comp_tel),
		downsize(tfm.star, n_comp_star),
		tfm.rv, tfm.reg_tel, tfm.reg_star, tfm.lih_t2b, tfm.lih_b2t,
		tfm.lih_o2b, tfm.lih_b2o, tfm.lih_t2o, tfm.lih_o2t, tfm.todo,
		tfm.instrument, tfm.order)

tel_prior(tfom::TFOrderModel) = model_prior(tfom.tel.lm, tfom.reg_tel)
star_prior(tfom::TFOrderModel) = model_prior(tfom.star.lm, tfom.reg_star)

spectra_interp(og_vals::AbstractMatrix, lih) =
    (og_vals[lih.li] .* (1 .- lih.ratios)) + (og_vals[lih.li .+ 1] .* (lih.ratios))

tel_model(tfom) = spectra_interp(tfom.tel.lm(), tfom.lih_t2o)
star_model(tfom) = spectra_interp(tfom.star.lm(), tfom.lih_b2o)
rv_model(tfom) = spectra_interp(tfom.rv.lm(), tfom.lih_b2o)


function fix_FullLinearModel_s!(flm, min::Number, max::Number)
	@assert all(min .< flm.μ .< max)
	result = ones(typeof(flm.μ[1]), length(flm.μ))
	for i in 1:size(flm.s, 2)
		result[:] = _eval_full_lm(flm.M, flm.s[:, i], flm.μ)
		while any(result .> max) || any(result .< min)
			# println("$i, old s: $(lm.s[:, i]), min: $(minimum(result)), max:  $(maximum(result))")
			flm.s[:, i] ./= 2
			result[:] = _eval_full_lm(flm.M, flm.s[:, i], flm.μ)
			# println("$i, new s: $(lm.s[:, i]), min: $(minimum(result)), max:  $(maximum(result))")
		end
	end
end

function get_marginal_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::AbstractVector,
    xs::AbstractVector)
    gp_post = posterior(finite_GP, ys)
    gpx_post = gp_post(xs)
    return TemporalGPs.marginals(gpx_post)
end

function get_mean_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::AbstractVector,
    xs::AbstractVector)
    return mean.(get_marginal_GP(finite_GP, ys, xs))
end

function build_gp(θ)
    σ², l = θ
    k = σ² * TemporalGPs.stretch(TemporalGPs.Matern52(), 1 / l)
    f_naive = TemporalGPs.GP(k, TemporalGPs.GPC())
    f = to_sde(f_naive, SArrayStorage(Float64))
    #f = to_sde(f_naive)   # if develop issues with StaticArrays could revert to this
end

SOAP_gp = build_gp([3.3270754364467443, 9.021560480866474e-5])


function _spectra_interp_gp!(fluxes, vars, log_λ, flux_obs, var_obs, log_λ_obs; gp_mean::Number=1.)
	for i in 1:size(flux_obs, 2)
		gp = get_marginal_GP(SOAP_gp(log_λ_obs[:, i], var_obs[:, i]), flux_obs[:, i] .- gp_mean, log_λ)
		fluxes[:, i] = mean.(gp) .+ gp_mean
        vars[:, i] = var.(gp)
	end
end

function _spectra_interp_gp_div_gp!(fluxes::AbstractMatrix, vars::AbstractMatrix, log_λ::AbstractVector, flux_obs::AbstractMatrix, var_obs::AbstractMatrix, log_λ_obs::AbstractMatrix, flux_other::AbstractMatrix, var_other::AbstractMatrix, log_λ_other::AbstractMatrix; gp_mean::Number=1.)
	for i in 1:size(flux_obs, 2)
		gpn = get_marginal_GP(SOAP_gp(log_λ_obs[:, i], var_obs[:, i]), flux_obs[:, i] .- gp_mean, log_λ)
		gpd = get_marginal_GP(SOAP_gp(log_λ_other[:, i], var_other[:, i]), flux_other[:, i] .- gp_mean, log_λ)
		gpn_μ = mean.(gpn) .+ gp_mean
		gpd_μ = mean.(gpd) .+ gp_mean
		fluxes[:, i] = gpn_μ ./ gpd_μ
        vars[:, i] = (var.(gpn) .+ ((gpn_μ .^ 2 .* var.(gpd)) ./ (gpd_μ .^2))) ./ (gpd_μ .^2)
	end
end

function n_comps_needed(tfsm::TFSubmodel; threshold::Real=0.05)
    @assert 0 < threshold < 1
    s_var = sum(abs2, tfsm.lm.s; dims=2)
    return findfirst(s_var ./ sum(s_var) .< threshold)[1] - 1
end

function initialize!(tfom::TFOrderModel, tfd::TFData; min::Number=0, max::Number=1.2, use_gp::Bool=false, kwargs...)

	μ_min = min + 0.05
	μ_max = max - 0.05

	n_obs = size(tfd.flux, 2)
	n_comp_star = size(tfom.star.lm.M, 2) + 1
	n_comp_tel = size(tfom.tel.lm.M, 2)

	if use_gp
		star_log_λ_tel = _shift_log_λ_model(tfd.log_λ_obs, tfd.log_λ_star, tfom.star.log_λ)
		tel_log_λ_star = _shift_log_λ_model(tfd.log_λ_star, tfd.log_λ_obs, tfom.tel.log_λ)
		flux_star = ones(length(tfom.star.log_λ), n_obs)
		vars_star = ones(length(tfom.star.log_λ), n_obs)
		flux_tel = ones(length(tfom.tel.log_λ), n_obs)
		vars_tel = ones(length(tfom.tel.log_λ), n_obs)
		_spectra_interp_gp!(flux_star, vars_star, tfom.star.log_λ, tfd.flux, tfd.var, tfd.log_λ_star)
	else
		flux_star = spectra_interp(tfd.flux, tfom.lih_o2b)
	end
	tfom.star.lm.μ[:] = make_template(flux_star; min=μ_min, max=μ_max)
	_, _, _, _, rvs_naive =
	    DPCA(flux_star, tfom.star.λ; template=tfom.star.lm.μ, num_components=1)

	# telluric model with stellar template
	if use_gp
		flux_star .= tfom.star.lm.μ
		_spectra_interp_gp_div_gp!(flux_tel, vars_tel, tfom.tel.log_λ, tfd.flux, tfd.var, tfd.log_λ_obs, flux_star, vars_star, star_log_λ_tel)
	else
		flux_tel = spectra_interp(tfd.flux, tfom.lih_o2t) ./
			spectra_interp(repeat(tfom.star.lm.μ, 1, n_obs), tfom.lih_b2t)
	end
	tfom.tel.lm.μ[:] = make_template(flux_tel; min=μ_min, max=μ_max)
	# _, tfom.tel.lm.M[:, :], tfom.tel.lm.s[:, :], _ =
	#     fit_gen_pca(flux_tel; num_components=n_comp_tel, mu=tfom.tel.lm.μ)

	# stellar model with telluric template
	if use_gp
		flux_tel .= tfom.tel.lm.μ
		_spectra_interp_gp_div_gp!(flux_star, vars_star, tfom.star.log_λ, tfd.flux, tfd.var, tfd.log_λ_star, flux_tel, vars_tel, tel_log_λ_star)
	else
		_tel_μ = spectra_interp(repeat(tfom.tel.lm.μ, 1, n_obs), tfom.lih_t2b)
		flux_star = spectra_interp(tfd.flux, tfom.lih_o2b) ./ _tel_μ
		vars_star = spectra_interp(tfd.var, tfom.lih_o2b) ./ _tel_μ  # TODO:should be adding in quadtrature, but fix later
	end
	tfom.star.lm.μ[:] = make_template(flux_star; min=μ_min, max=μ_max)
	_, M_star, s_star, rvs_notel =
	    DEMPCA(flux_star, tfom.star.λ, 1 ./ vars_star; template=tfom.star.lm.μ, num_components=n_comp_star)

	# telluric model with updated stellar template
	if use_gp
		flux_star .= tfom.star.lm.μ
		_spectra_interp_gp_div_gp!(flux_tel, vars_tel, tfom.tel.log_λ, tfd.flux, tfd.var, tfd.log_λ_obs, flux_star, vars_star, star_log_λ_tel)
	else
		_star_μ = spectra_interp(repeat(tfom.star.lm.μ, 1, n_obs), tfom.lih_b2t)
		flux_tel = spectra_interp(tfd.flux, tfom.lih_o2t) ./ _star_μ
		vars_tel = spectra_interp(tfd.var, tfom.lih_o2t) ./ _star_μ  # TODO:should be adding in quadtrature, but fix later
	end
	tfom.tel.lm.μ[:] = make_template(flux_tel; min=μ_min, max=μ_max)
	m_tel = empca.empca((flux_tel .- tfom.tel.lm.μ)', 1 ./ vars_tel', nvec=n_comp_tel, silent=true)
	tfom.tel.lm.M[:, :] = m_tel.eigvec'
	tfom.tel.lm.s[:, :] = m_tel.coeff'

	tfom.star.lm.M[:, :], tfom.star.lm.s[:] = M_star[:, 2:end], s_star[2:end, :]
	tfom.rv.lm.M[:, :], tfom.rv.lm.s[:] = M_star[:, 1], s_star[1, :]'

	fix_FullLinearModel_s!(tfom.star.lm, min, max)
	fix_FullLinearModel_s!(tfom.tel.lm, min, max)

	return rvs_notel, rvs_naive
end

L1(a::AbstractArray) = sum(abs.(a))
L2(a::AbstractArray) = sum(a .* a)
shared_attention(v1::AbstractVector, v2::AbstractVector) = dot(abs.(v1), abs.(v2))

function model_prior(lm, reg::Dict{Symbol, <:Real})
	val = 0
	if haskey(reg, :shared_M)
		shared_att = 0
		for i in size(lm.M, 2)
			for j in size(lm.M, 2)
				if i != j
					shared_att += shared_attention(lm.M[:, i], lm.M[:, j])
				end
			end
		end
		val += shared_att * reg[:shared_M]
	end
	if haskey(reg, :L2_μ) || haskey(reg, :L1_μ) || haskey(reg, :L1_μ₊_factor)
		μ_mod = lm.μ .- 1
		if haskey(reg, :L2_μ); val += L2(μ_mod) * reg[:L2_μ] end
		if haskey(reg, :L1_μ)
			val += L1(μ_mod) * reg[:L1_μ]
			if haskey(reg, :L1_μ₊_factor)
				val += sum(view(μ_mod, μ_mod .> 0)) * reg[:L1_μ₊_factor] * reg[:L1_μ]
			end
		end
	end
	if haskey(reg, :L2_M); val += L2(lm.M) * reg[:L2_M] end
	if haskey(reg, :L1_M); val += L1(lm.M) * reg[:L1_M] end
	if haskey(reg, :L1_M) || haskey(reg, :L2_M); val += L1(lm.s) end
	return val
end

struct TFOutput{T<:Real}
	tel::AbstractMatrix{T}
	star::AbstractMatrix{T}
	rv::AbstractMatrix{T}
	TFOutput(tfom::TFOrderModel) = TFOutput(tel_model(tfom), star_model(tfom), rv_model(tfom))
	function TFOutput(tel::AbstractMatrix{T}, star, rv) where {T<:Real}
		@assert size(tel) == size(star) == size(rv)
		new{T}(tel, star, rv)
	end
end
