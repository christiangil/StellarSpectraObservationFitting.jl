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

function lower_inds_and_ratios(template_λs, obs_λs::Matrix)
    len_obs, n_obs = size(obs_λs)
    lower_inds = zeros(Int, (len_obs, n_obs))
    ratios = zeros((len_obs, n_obs))
    len_template = length(template_λs)
    for j in 1:n_obs
        current_λs = view(obs_λs, :, j)
        lower_inds[:, j] = searchsortednearest(template_λs, current_λs; lower=true)
        for i in 1:len_obs
            x0 = template_λs[lower_inds[i, j]]
            x1 = template_λs[lower_inds[i, j]+1]
            x = current_λs[i]
            ratios[i, j] = (x - x0) / (x1 - x0)
        end
        lower_inds[:, j] .+= (j - 1) * len_template
    end
    return lower_inds, ratios
end
