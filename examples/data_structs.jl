using JLD2
using Stheno
using TemporalGPs
using UnitfulAstro, Unitful

struct SpectraHolder{T<:AbstractArray{<:Real,1}}
    log_λ_obs::T
    log_λ_bary::T
    flux_obs::T
    var_obs::T
    function SpectraHolder(
        log_λ_obs::T,
        log_λ_bary::T,
        flux_obs::T,
        var_obs::T,
    ) where {T<:AbstractArray{<:Real,1}}
        @assert 1 <=
                length(log_λ_obs) ==
                length(log_λ_bary) ==
                length(flux_obs) ==
                length(var_obs)
        new{typeof(log_λ_obs)}(log_λ_obs, log_λ_bary, flux_obs, var_obs)
    end
end
