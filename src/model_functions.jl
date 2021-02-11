function create_λ_template(log_λ_obs, resolution)
    log_min_wav, log_max_wav = [minimum(log_λ_obs), maximum(log_λ_obs)]
    len = Int(ceil((exp(log_max_wav) - exp(log_min_wav)) * resolution / exp((log_max_wav + log_min_wav)/2)))
    log_Δλ = (log(log_max_wav) - log(log_min_wav)) / len
    len += 2
    log_λ_template = range(log_min_wav - log_Δλ; length = len,  stop = log_max_wav + log_Δλ)
    λ_template = exp.(log_λ_template)
    return len, log_λ_template, λ_template
end

struct LinearInterpolationHelper{T<:Number}
    li::Matrix{Int}  # lower indices
    ratios::Matrix{T}
	function LinearInterpolationHelper(
		li::Matrix{Int},  # lower indices
	    ratios::Matrix{T}) where {T<:Number}
		@assert size(li) == size(ratios)
		@assert all(0 .<= ratios .<= 1)
		# @assert some issorted thing?
		return new{T}(li, ratios)
	end
end

function LinearInterpolationHelper_maker(to_λs, from_λs::Matrix)
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
	M::Array{T}
	s::Array{T}
	μ::Vector{T}
	function FullLinearModel(M, s, μ)
		@assert length(μ) == size(M, 1)
		@assert size(M, 2) == size(s, 1)
		return new{T}(M, s, μ)
	end
end
# Base (no mean) linear model
struct BaseLinearModel{T<:Number} <: LinearModel
	M::Array{T}
	s::Array{T}
	function BaseLinearModel(M, s)
		@assert size(M, 2) == size(s, 1)
		return new{T}(M, s)
	end
end
_eval_lm(lm::LinearModel) = lm.M * lm.s
(lm::BaseLinearModel)() = _eval_lm(lm)
(lm::FullLinearModel)() = _eval_lm(lm) .+ lm.μ


struct TFSubmodel{T<:Number}
    len::Int
    log_λ::AbstractVector{T}
    λ::AbstractVector{T}
    flux::Matrix{T}
	lm::LinearModel{T}
    function TFSubmodel(log_λ_obs, model_res, n_comp)
        n_obs = size(log_λ_obs, 2)
		len, log_λ, λ = create_λ_template(log_λ_obs, model_res)
        flux = ones(len, n_obs)
		lm = FullLinearModel(zeros(len, n_comp), zeros(n_comp, n_obs), ones(len))
        return TFSubmodel(len, log_λ, λ, flux, lm)
    end
    function TFSubmodel(
        len::Int,
        log_λ::AbstractVector{T},
        λ::AbstractVector{T},
        flux::Matrix{T},
		lm::LinearModel{T}) where {T<:Number}
        @assert len == length(log_λ) == length(λ) == size(flux, 1) == length(lm.μ) == size(lm.M, 1)
		@assert size(flux, 2) == size(lm.s, 2)
        return new{T}(len, log_λ, λ, flux, lm)
    end
end

struct TFModel{T<:Number}
    tel::TFSubmodel{T}
    star::TFSubmodel{T}
	rv::TFSubmodel{T}
    lih_t2b::LinearInterpolationHelper
    lih_b2t::LinearInterpolationHelper
    lih_o2b::LinearInterpolationHelper
    lih_b2o::LinearInterpolationHelper
    lih_t2o::LinearInterpolationHelper
    lih_o2t::LinearInterpolationHelper
    function TFModel(
		log_λ_obs::Matrix,
		log_λ_obs_bary::Matrix,
		star_model_res::Number,
		tel_model_res::Number;
		n_comp_tel::Int=2,
		n_comp_star::Int=2)

        tel = TFSubmodel(log_λ_obs, tel_model_res, n_comp_tel)
        star = TFSubmodel(log_λ_obs_bary, star_model_res, n_comp_star)
		rv = TFSubmodel(log_λ_obs_bary, star_model_res, 1)

        n_obs = size(log_λ_obs, 2)
        lih_t2o, lih_o2t = LinearInterpolationHelper_maker(tel.log_λ, log_λ_obs)
        lih_b2o, lih_o2b = LinearInterpolationHelper_maker(star.log_λ, log_λ_obs_bary)
        star_dop = [log_λ_obs_bary[1, i] - log_λ_obs[1, i] for i in 1:n_obs]
        star_log_λ_tel = ones(size(star.flux))
        for i in 1:n_obs
            star_log_λ_tel[:, i] = star.log_λ .+ star_dop[i]
        end
        lih_t2b, lih_b2t = LinearInterpolationHelper_maker(tel.log_λ, star_log_λ_tel)
        return TFModel(tel, star, rv, lih_t2b, lih_b2t, lih_o2b, lih_b2o, lih_t2o, lih_o2t)
    end
    TFModel(tel::TFSubmodel{T}, star, rv, lih_t2b, lih_b2t, lih_o2b, lih_b2o,
		lih_t2o, lih_o2t) where {T<:Number} =
		new{T}(tel, star, rv, lih_t2b, lih_b2t, lih_o2b, lih_b2o, lih_t2o, lih_o2t)
end

spectra_interp(og_vals::Matrix, lih) =
    (og_vals[lih.li] .* (1 .- lih.ratios)) + (og_vals[lih.li .+ 1] .* (lih.ratios))

function initialize!(tfm, flux_obs::Matrix)

	n_obs = size(flux_obs, 1)

	tfm.star.flux[:, :] = spectra_interp(flux_obs, tfm.lih_o2b)
	tfm.star.lm.μ[:] = make_template(tfm.star.flux)
	_, tfm.star.lm.M[:, :], tfm.star.lm.s[:, :], _, rvs_notel =
	    DPCA(tfm.star.flux, tfm.star.λ; template=tfm.star.lm.μ)

	rvs_naive = copy(rvs_notel)

	# telluric model with stellar template
	tfm.tel.flux[:, :] = spectra_interp(flux_obs, tfm.lih_o2t) ./
		spectra_interp(repeat(tfm.star.lm.μ, 1, n_obs), tfm.lih_b2t)
	tfm.tel.lm.μ[:] = make_template(tfm.tel.flux)
	_, tfm.tel.lm.M[:, :], tfm.tel.lm.s[:, :], _ =
	    fit_gen_pca(tfm.tel.flux; num_components=2, mu=tfm.tel.lm.μ)

	# stellar model with telluric template
	tfm.star.flux[:, :] = spectra_interp(flux_obs, tfm.lih_o2b) ./
		spectra_interp(repeat(tfm.tel.lm.μ, 1, n_obs), tfm.lih_t2b)
	tfm.star.lm.μ[:] = make_template(tfm.star.flux)
	_,tfm.star.lm.M[:, :], tfm.star.lm.s[:, :], _, rvs_notel[:] =
	    DPCA(tfm.star.flux, tfm.star.λ; template = tfm.star.lm.μ)

	# telluric model with updated stellar template
	tfm.tel.flux[:, :] = spectra_interp(flux_obs, tfm.lih_o2t) ./
		spectra_interp(repeat(tfm.star.lm.μ, 1, n_obs), tfm.lih_b2t)
	tfm.tel.lm.μ[:] = make_template(tfm.tel.flux)
	_, tfm.tel.lm.M[:, :], tfm.tel.lm.s[:, :], _ =
	    fit_gen_pca(tfm.tel.flux; num_components=2, mu=tfm.tel.lm.μ)

	return rvs_notel, rvs_naive
end


L1(thing) = sum(abs.(thing))
L2(thing) = sum(thing .* thing)

function model_prior(lm::FullLinearModel, coeffs::Vector{<:Real})
    μ_mod = lm.μ .- 1
    return (coeffs[3] * L2(μ_mod)) +
	(coeffs[2] * (L1(μ_mod) + (coeffs[1] * sum(μ_mod[μ_mod.>0])))) +
	(coeffs[5] * L2(lm.M)) +
    (coeffs[4] * L1(lm.M)) +
    L1(lm.s)
end

tel_model(tfm::TFModel) = spectra_interp(tfm.tel.lm(), tfm.lih_t2o)
star_model(tfm::TFModel) = spectra_interp(tfm.star.lm(), tfm.lih_b2o)
rv_model(tfm::TFModel) = spectra_interp(tfm.star.lm(), tfm.lih_b2o)
