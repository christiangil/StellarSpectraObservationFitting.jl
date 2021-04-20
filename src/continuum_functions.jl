using GPLinearODEMaker; GLOM = GPLinearODEMaker
using LinearAlgebra

function vander(x::Vector{T}, n::Int) where {T <: Number}
    m = ones(T, length(x), n + 1)
    for i in 1:n
        m[:, i + 1] = m[:, i] .* x
    end
    return m
end

"""
Solve a linear system of equations (optionally with variance values at each point or covariance array)
see (https://en.wikipedia.org/wiki/Generalized_least_squares#Method_outline)
"""
function general_lst_sq(
    dm::Matrix{T},
    data::Vector,
    Σ::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}}) where {T<:Real}
    if ndims(Σ) == 1
        Σ = Diagonal(Σ)
    else
        Σ = GLOM.ridge_chol(Σ)
    end
    return (dm' * (Σ \ dm)) \ (dm' * (Σ \ data))
end

function fit_continuum(x::Vector, y::Vector, σ²::Vector; order::Int=6, nsigma::Vector{<:Real}=[0.3,3.0], maxniter::Int=50, plot_stuff::Bool=false)
    """Fit the continuum using sigma clipping
    Args:
        x: The wavelengths
        y: The log-fluxes
        ivars : inverse variances for `ys`.
        order: The polynomial order to use
        nsigma: The sigma clipping threshold: tuple (low, high)
        maxniter: The maximum number of iterations to do
    Returns:
        The value of the continuum at the wavelengths in x
    """
    @assert 0 <= order < length(x)
    @assert length(x) == length(y) == length(σ²)
    @assert length(nsigma) == 2

    A = vander(x .- mean(x), order)
    m = fill(true, length(x))
    μ = ones(length(x))
    for i in 1:maxniter
        m[σ² .== Inf] .= false  # mask out the bad pixels
        w = general_lst_sq(A[m, :], y[m], σ²[m])
        μ[:] = A * w
        resid = y - μ
        sigma = median(abs.(resid))
        m_new = (-nsigma[1]*sigma) .< resid .< (nsigma[2]*sigma)
        if sum(m) == sum(m_new); break end
        m = m_new
    end
    return μ
end
