using UnitfulAstro, Unitful

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


make_template(matrix::Matrix{T}) where {T<:Real} = vec(median(matrix, dims=2))
make_template_mean(matrix::Matrix{T}) where {T<:Real} =
    vec(mean(matrix, dims=2))

function shift_log_λ(v::Unitful.Velocity, log_λ::Vector{T}) where {T<:Real}
    return log_λ .+ (log((1.0 + v / light_speed) / (1.0 - v / light_speed)) / 2)
end
