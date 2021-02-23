using UnitfulAstro, Unitful
using Statistics

light_speed = uconvert(u"m/s", 1u"c")
light_speed_nu = ustrip(light_speed)

function searchsortednearest(a::AbstractVector{T} where T<:Real, x::Real; lower::Bool=false)
   	idx = searchsortedfirst(a,x)
   	if (idx==1); return idx; end
   	if (idx>length(a)); return length(a); end
   	if (a[idx]==x); return idx; end
   	if lower || ((abs(a[idx]-x) >= abs(a[idx-1]-x)))
      	return idx - 1
   	else
      	return idx
   	end
end


function searchsortednearest(a::AbstractVector{T} where T<:Real, x::AbstractVector{T} where T<:Real; lower::Bool=false)
   	len_x = length(x)
   	len_a = length(a)
   	idxs = zeros(Int64, len_x)
   	idxs[1] = searchsortednearest(a, x[1]; lower=lower)
	for i in 2:len_x
	   	idxs[i] = idxs[i-1] + searchsortednearest(view(a, idxs[i-1]:len_a), x[i]; lower=lower) - 1
   	end
   	return idxs
end

function clip_vector!(vec::Vector; max::Number=Inf, min::Number=-Inf)
	vec[vec .< min] .= min
	vec[vec .> max] .= max
end

function make_template(matrix::Matrix; use_mean::Bool=true, kwargs...)
	if use_mean
		result = vec(median(matrix, dims=2))
	else
		result = vec(mean(matrix, dims=2))
	end
	clip_vector!(result; kwargs...)
	return result
end

function shift_log_λ(v::Unitful.Velocity, log_λ::Vector{T}) where {T<:Real}
	return log_λ .+ (log((1.0 + v / light_speed) / (1.0 - v / light_speed)) / 2)
end
