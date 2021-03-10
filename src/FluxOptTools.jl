# Shamelessly crimped (with some modifications) from
# https://github.com/baggepinnen/FluxOptTools.jl

using LinearAlgebra, Optim, Flux, Zygote# , RecipesBase
import Base.copyto!

veclength(grads::Zygote.Grads) = sum(length(g[1]) for g in grads.grads if !(typeof(g[1])<:GlobalRef))
veclength(params::Flux.Params) = sum(length, params.params)
veclength(x) = length(x)
Base.zeros(grads::Zygote.Grads) = zeros(veclength(grads))
Base.zeros(pars::Flux.Params) = zeros(veclength(pars))

# Grads =============================================

function copyto!(v::AbstractArray, grads::Zygote.Grads)
    @assert length(v) == veclength(grads)
    s = 1
    for g in grads.grads
        if !(typeof(g[1])<:GlobalRef)
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


function optfuns(loss, pars::Union{Flux.Params, Zygote.Params})
    grads = Zygote.gradient(loss, pars)
    p0 = copyto!(zeros(pars), pars)
    gradfun = function (g,w)
        copyto!(pars, w)
        grads = Zygote.gradient(loss, pars)
        copyto!(g, grads)
    end
    lossfun = function (w)
        copyto!(pars, w)
        loss()
    end
    fg! = function (F,G,w)
        copyto!(pars, w)
        if G != nothing
            l, back = Zygote.pullback(loss, pars)
            grads = back(1)
            copyto!(G, grads)
            return l
        end
        if F != nothing
            return loss()
        end
    end
    lossfun, gradfun, fg!, p0
end


## telfitting additions

_loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, tfd::TFData) =
    sum((((tel .* (star + rv)) - tfd.flux) .^ 2) ./ tfd.var)
loss(tfo::TFOutput, tfd) = _loss(tfo.tel, tfo.star, tfo.rv, tfd)
loss(tfm::TFModel, tfd) = _loss(tel_model(tfm), star_model(tfm), rv_model(tfm), tfd)
loss_tel(tfo::TFOutput, tfm::TFModel, tfd) = _loss(tel_model(tfm), tfo.star, tfo.rv, tfd) + tel_prior(tfm)
loss_star(tfo::TFOutput, tfm::TFModel, tfd) = _loss(tfo.tel, star_model(tfm), tfo.rv, tfd) + star_prior(tfm)
loss_rv(tfo::TFOutput, tfm::TFModel, tfd) = _loss(tfo.tel, tfo.star, rv_model(tfm), tfd)
function loss_funcs(tfo::TFOutput, tfm::TFModel, tfd::TFData)
    l() = loss(tfo, tfd)
    l_tel() = loss_tel(tfo, tfm, tfd)
    l_star() = loss_star(tfo, tfm, tfd)
    l_rv() = loss_rv(tfo, tfm, tfd)
    return l, l_tel, l_star, l_rv
end


struct TFOptimSubWorkspace
    θ::Flux.Params
    fg!::Function
    p0::Vector
    function TFOptimSubWorkspace(tfsm::TFSubmodel, loss::Function; only_s::Bool=false)
        if only_s
            θ = Flux.params(tfsm.lm.s)
        else
            θ = Flux.params(tfsm.lm.M, tfsm.lm.s, tfsm.lm.μ)
        end
        _, _, fg!, p0 = optfuns(loss, θ)
        return TFOptimSubWorkspace(θ, fg!, p0)
    end
    function TFOptimSubWorkspace(θ, fg!, p0)
        len = 0
        for i in 1:length(θ)
            len += length(θ[i])
        end
        @assert len == length(p0)
        new(θ, fg!, p0)
    end
end

struct TFOptimWorkspace
    tel::TFOptimSubWorkspace
    star::TFOptimSubWorkspace
    rv::TFOptimSubWorkspace
    tfm::TFModel
    tfo::TFOutput
    tfd::TFData
    function TFOptimWorkspace(tfm::TFModel, tfo::TFOutput, tfd::TFData; return_loss_f::Bool=false)
        loss, loss_tel, loss_star, loss_rv = loss_funcs(tfo, tfm, tfd)
        tel = TFOptimSubWorkspace(tfm.tel, loss_tel; only_s=false)
        star = TFOptimSubWorkspace(tfm.star, loss_star; only_s=false)
        rv = TFOptimSubWorkspace(tfm.rv, loss_rv; only_s=true)
        tfow = TFOptimWorkspace(tel, star, rv, tfm, tfo, tfd)
        if return_loss_f
            return tfow, loss
        else
            return tfow
        end
    end
    function TFOptimWorkspace(tel, star, rv, tfm, tfo, tfd)
        @assert length(tel.θ) == length(star.θ) == 3
        @assert length(rv.θ) == 1
        new(tel, star, rv, tfm, tfo, tfd)
    end
end

function _Flux_optimize!(fg!::Function, p0::Vector, θ::Flux.Params, OOptions::Optim.Options)
    Optim.optimize(Optim.only_fg!(fg!), p0, OOptions)
    copyto!(p0, θ)
end
_Flux_optimize!(tfosw::TFOptimSubWorkspace, OOptions) =
    _Flux_optimize!(tfosw.fg!, tfosw.p0, tfosw.θ, OOptions)

function train_TFModel!(tfow::TFOptimWorkspace; OOptions::Optim.Options=Optim.Options(iterations=10, f_tol=1e-3, g_tol=1e5))
    # optimize star
    _Flux_optimize!(tfow.star, OOptions)
    tfow.tfo.star[:, :] = star_model(tfow.tfm)

    # optimize RVs
    tfow.tfm.rv.lm.M[:] = calc_doppler_component_RVSKL(tfow.tfm.star.λ, tfow.tfm.star.lm.μ)
    _Flux_optimize!(tfow.rv, OOptions)
    tfow.tfo.rv[:, :] = rv_model(tfow.tfm)

    # optimize tellurics
    _Flux_optimize!(tfow.tel, OOptions)
    tfow.tfo.star[:, :] = tel_model(tfow.tfm)
end
