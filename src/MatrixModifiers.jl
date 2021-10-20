using SparseArrays
import Base.copy
import Base.*

abstract type MatrixModifier end

struct LinearInterpolator <: MatrixModifier
    li::AbstractMatrix{Int}  # lower indices
    ratios::AbstractMatrix
	function LinearInterpolator(
		li::AbstractMatrix{Int},  # lower indices
	    ratios::AbstractMatrix)
		@assert size(li) == size(ratios)
		@assert all(0 .<= ratios .<= 1)
		# @assert some issorted thing?
		return new(li, ratios)
	end
end
Base.copy(d::LinearInterpolator) = LinearInterpolator(copy(d.flux), copy(d.var), copy(d.log_λ_obs), copy(d.log_λ_star), copy(lsf_broadener))

function (lih::LinearInterpolator)(inds::AbstractVecOrMat, n_in::Int)
	n_out, n_obs = size(lih.li)
	@assert all(0 .< inds .<= n_obs)
	@assert allunique(inds)
	j = inds[1]-1
	new_li = ones(Int, n_out, length(inds))
	new_li[:, 1] = lih.li[:, inds[1]] .- (j * n_in)
	for i in 2:length(inds)
		j += (inds[i] - inds[i - 1]) - 1
		new_li[:, i] = lih.li[:, inds[i]] .- (j * n_in)
	end
	return LinearInterpolator(new_li, view(lih.ratios, :, inds))
end
function LinearInterpolator_maker(from_λs::AbstractVecOrMat, to_λs::AbstractMatrix)
	len_to, n_obs = size(to_λs)
	len_from = size(from_λs, 1)

	lower_inds = Array{Int64}(undef, len_to, n_obs)
	ratios = Array{Float64}(undef, len_to, n_obs)

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
		helper!(j, len_to, lower_inds, ratios, from_λs, view(to_λs, :, j))
		lower_inds[:, j] .+= (j - 1) * len_from
	end

	return LinearInterpolator(lower_inds, ratios)
end

function _oversampled_interpolation_ratios!(holder::AbstractVector, x_lo::Real, x_hi::Real, lo_ind::Int, hi_ind::Int, from_x::AbstractVector)
	if from_x[lo_ind] < x_lo; lo_ind += 1 end
	if from_x[hi_ind] > x_hi; hi_ind -= 1 end

	edge_term_lo = (from_x[lo_ind] - x_lo) ^ 2 / (from_x[lo_ind] - from_x[lo_ind-1])
	edge_term_hi = (x_hi - from_x[hi_ind]) ^ 2 / (from_x[hi_ind+1] - from_x[hi_ind])

	holder[1] = edge_term_lo
	holder[2] = from_x[lo_ind+1] + from_x[lo_ind] - 2 * x_lo - edge_term_lo

	holder[3:end-2] .= view(from_x, lo_ind+2:hi_ind) .- view(from_x, lo_ind:hi_ind-2)

	holder[end-1] = 2 * x_hi - from_x[hi_ind] - from_x[hi_ind-1] - edge_term_hi
	holder[end] = edge_term_hi

	@assert isapprox(sum(holder), 2 * (x_hi - x_lo))
	holder ./= sum(holder)
end


struct OversampledInterpolator <: MatrixModifier
    inds::AbstractMatrix{UnitRange}  # UnitRanges of applicabale pixels
    ratios::AbstractMatrix{AbstractVector}  # how much of each  pixel to use
	function OversampledInterpolator(
		inds::AbstractMatrix{UnitRange},  # lower indices
	    ratios::AbstractMatrix{AbstractVector})
		@assert size(inds) == size(ratios)
		@assert all([all(i .> 0) for i in ratios])
		@assert all([length(inds[i])==length(ratios[i]) for i in 1:length(inds)])
		return new(inds, ratios)
	end
end
function (ih::OversampledInterpolator)(inds::AbstractVecOrMat)
	@assert all(0 .< inds .<= size(ih.inds, 2))
	@assert allunique(inds)
	return OversampledInterpolator(view(ih.inds, :, inds), view(ih.ratios, :, inds))
end
function OversampledInterpolator_maker(from_x::AbstractVector, to_bounds::AbstractMatrix)
	inds = Matrix{UnitRange}(undef, size(to_bounds, 1) - 1, length(from_x))
    ratios = Matrix{Vector{Float64}}(undef, size(to_bounds, 1) - 1, length(from_x))
	for j in 1:size(to_bounds, 2)
		bounds_inds = searchsortednearest(from_x, view(to_bounds, :, j))
		for i in 1:size(inds, 1)
			x_lo, x_hi = to_bounds[i], to_bounds[i+1]
			lo_ind, hi_ind = bounds_inds[i], bounds_inds[i+1]
			if from_x[lo_ind] < x_lo; lo_ind += 1 end
			if from_x[hi_ind] > x_hi; hi_ind -= 1 end
			inds[i, j] = lo_ind-1:hi_ind+1
			ratios[i, j] = Array{Float64}(undef, length(inds[i, j]))
			_oversampled_interpolation_ratios!(ratios[i, j], x_lo, x_hi, lo_ind, hi_ind, from_x)
		end
	end
	return OversampledInterpolator(inds, ratios)
end


struct ConstantOversampledInterpolator <: MatrixModifier
    inds::AbstractVector{UnitRange}  # UnitRanges of applicabale pixels
    ratios::AbstractVector{<:AbstractArray}  # how much of each  pixel to use
	function ConstantOversampledInterpolator(
		inds::AbstractVector{UnitRange},  # lower indices
	    ratios::AbstractVector{<:AbstractArray})
		@assert size(inds) == size(ratios)
		@assert all([all(i .> 0) for i in ratios])
		@assert all([length(inds[i])==length(ratios[i]) for i in 1:length(inds)])
		return new(inds, ratios)
	end
end
(ih::ConstantOversampledInterpolator)(inds::AbstractVecOrMat) = ih
function ConstantOversampledInterpolator_maker(from_x::AbstractVector, to_bounds::AbstractVector)
	inds = Vector{UnitRange}(undef, length(to_bounds) - 1, length(from_x))
    ratios = Vector{Array{Float64}}(undef, length(to_bounds) - 1, length(from_x))
	bounds_inds = searchsortednearest(from_x, to_bounds)
	for i in 1:length(inds)
		x_lo, x_hi = to_bounds[i], to_bounds[i+1]
		lo_ind, hi_ind = bounds_inds[i], bounds_inds[i+1]
		if from_x[lo_ind] < x_lo; lo_ind += 1 end
		if from_x[hi_ind] > x_hi; hi_ind -= 1 end
		inds[i] = lo_ind-1:hi_ind+1
		ratios[i] = Array{Float64}(undef, 1, length(inds[i]))
		_oversampled_interpolation_ratios!(ratios[i], x_lo, x_hi, lo_ind, hi_ind, from_x)
	end
	return ConstantOversampledInterpolator(inds, ratios)
end


AcceptableMatrixModifier = Union{SparseMatrixCSC, Vector{SparseMatrixCSC}, MatrixModifier}
# (*)(ih::SparseMatrixCSC, model::AbstractMatrix) = ih * model
(*)(amm::Vector{SparseMatrixCSC}, model::AbstractMatrix) =
	hcat([amm[i] * view(model, :, i) for i in 1:size(model, 2)]...)
(*)(amm::LinearInterpolator, model::AbstractMatrix) =
	(model[amm.li] .* (1 .- amm.ratios)) + (model[amm.li .+ 1] .* (amm.ratios))
(*)(amm::OversampledInterpolator, model::AbstractMatrix) =
	hcat([[LinearAlgebra.BLAS.dot(view(model, amm.inds[i, j], j), amm.ratios[i, j]) for i in 1:length(amm.inds)] for j in 1:size(model, 2)]...)
(*)(amm::ConstantOversampledInterpolator, model::AbstractMatrix) =
	hcat([[LinearAlgebra.BLAS.dot(view(model, amm.inds[i], j), amm.ratios[i]) for i in 1:length(amm.inds)] for j in 1:size(model, 2)]...)
function _times_func(amm::ConstantOversampledInterpolator, model::AbstractMatrix)
	ans = Array{Float64}(undef, length(amm.inds), size(model, 2))
	for i in 1:length(amm.inds)
		for j in 1:size(model, 2)
			ans[i, j] = LinearAlgebra.BLAS.dot(view(model, amm.inds[i], j), amm.ratios[i])
		end
	end
	return ans
end
