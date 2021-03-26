# Shamelessly crimped (with some modifications) from
# https://github.com/baggepinnen/FluxOptTools.jl

using LinearAlgebra, Optim, Flux, Zygote# , RecipesBase
import Base.copyto!

veclength(grads::Zygote.Grads) = sum(length(g[1]) for g in grads.grads if !(typeof(g[1])<:GlobalRef || typeof(g[1])<:Dict))
veclength(params::Flux.Params) = sum(length, params.params)
veclength(x) = length(x)
Base.zeros(grads::Zygote.Grads) = zeros(veclength(grads))
Base.zeros(pars::Flux.Params) = zeros(veclength(pars))

# Grads =============================================

function copyto!(v::AbstractArray, grads::Zygote.Grads)
    @assert length(v) == veclength(grads)
    s = 1
    for g in grads.grads
        if !(typeof(g[1])<:GlobalRef || typeof(g[1])<:Dict)
            l = length(g[2])
            v[s:s+l-1] .= vec(g[2])
            s += l
        end
    end
    v
end

function copyto!(grads::Zygote.Grads, v::AbstractArray)
    s = 1
    for g in grads.grads
        l = length(g[2])
        g[2] .= reshape(v[s:s+l-1], size(g[2]))
        s += l
    end
    grads
end

# Params =============================================

function copyto!(v::AbstractArray, pars::Flux.Params)
    @assert length(v) == veclength(pars)
    s = 1
    for g in pars.params
        l = length(g)
        v[s:s+l-1] .= vec(g)
        s += l
    end
    v
end

function copyto!(pars::Flux.Params, v::AbstractArray)
    s = 1
    for p in pars.params
        l = length(p)
        p .= reshape(v[s:s+l-1], size(p))
        s += l
    end
    pars
end


function opt_funcs(loss, pars::Union{Flux.Params, Zygote.Params}; return_OD::Bool=true)
    p0 = copyto!(zeros(pars), pars)
    function g!(G, w)
        copyto!(pars, w)
        copyto!(G, Zygote.gradient(loss, pars))
    end
    function f(w)
        copyto!(pars, w)
        return loss()
    end
    function fg!(F, G, w)
        copyto!(pars, w)
        if G != nothing
            l, back = Zygote.pullback(loss, pars)
            copyto!(G, back(1))
            return l
        end
        if F != nothing
            return loss()
        end
    end
    function fg_obj!(G, w)
        copyto!(pars, w)
        l, back = Zygote.pullback(loss, pars)
        copyto!(G, back(1))
        return l
    end
    if return_OD
        return f, g!, fg!, p0, OnceDifferentiable(f, g!, fg_obj!, p0)
    else
        return f, g!, fg!, p0
    end
end
