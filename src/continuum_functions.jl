# Heavily inspired (if not blatantly ripped of from) the functions at
# https://github.com/megbedell/wobble/blob/master/wobble/data.py
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
    dm::AbstractMatrix{T},
    data::AbstractVector,
    Σ::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},AbstractMatrix{T},AbstractVector{T}}) where {T<:Real}
    if ndims(Σ) == 1
        Σ = Diagonal(Σ)
    else
        Σ = GLOM.ridge_chol(Σ)
    end
    return (dm' * (Σ \ dm)) \ (dm' * (Σ \ data))
end


# These were initially defined to act on all of the orders of the spectra at a
# given time, but I have defined them to act on all of the times of the spectra
# at a given order. Should be equivalent
function fit_continuum(x::AbstractVector, y::AbstractVector, σ²::AbstractVector; order::Int=6, nsigma::Vector{<:Real}=[0.3,3.0], maxniter::Int=50, plot_stuff::Bool=false)
    """Fit the continuum using sigma clipping
    Args:
        x: The wavelengths
        y: The log-fluxes
        σ² : variances for `ys`.
        order: The polynomial order to use
        nsigma: The sigma clipping threshold: vector (low, high)
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
        w = general_lst_sq(view(A, m, :), view(y, m), view(σ², m))
        μ[:] = A * w
        resid = y - μ
        sigma = median(abs.(resid))
        m_new = (-nsigma[1]*sigma) .< resid .< (nsigma[2]*sigma)
        if sum(m) == sum(m_new); break end
        m = m_new
    end
    return μ
end
function continuum_normalize!(tfd; kwargs...)
	continuum = ones(size(tfd.log_λ_obs, 1))
	for i in 1:size(tfd.log_λ_obs, 2)
		continuum[:] = fit_continuum(view(tfd.log_λ_obs, :, i), view(tfd.flux, :, i), view(tfd.var, :, i); kwargs...)
		tfd.flux[:, i] ./= continuum
		tfd.var[:, i] ./= continuum .* continuum
	end
end

function mask_low_pixels!(y::AbstractVector, σ²::AbstractVector; min_flux::Real= 0., padding::Int= 2)
	bad = y .< min_flux
	for i in 1:length(bad)
		bad[i] = bad[i] || !isfinite(y[i])
	end
	y[bad] .= min_flux
	l = length(bad)
	for i in findall(bad)
		bad[max(1, i - padding):min(i - padding, l)] .= true
	end
	σ²[bad] .= Inf
end
function mask_low_pixels!(tfd; kwargs...)
	for i in 1:size(tfd.log_λ_obs, 2)
		mask_low_pixels!(view(tfd.flux, :, i), view(tfd.var, :, i); kwargs...)
	end
end

function mask_high_pixels!(y::AbstractVector, σ²::AbstractVector; max_flux::Real= 2., padding::Int= 2)
	bad = y .> max_flux
	for i in 1:length(bad)
		bad[i] = bad[i] || !isfinite(y[i])
	end
	y[bad] .= max_flux
	l = length(bad)
	for i in findall(bad)
		bad[max(1, i - padding):min(i - padding, l)] .= true
	end
	σ²[bad] .= Inf
end
function mask_high_pixels!(tfd; kwargs...)
	for i in 1:size(tfd.log_λ_obs, 2)
		mask_high_pixels!(view(tfd.flux, :, i), view(tfd.var, :, i); kwargs...)
	end
end

function mask_bad_edges!(y::AbstractVector, σ²::AbstractVector; window_width::Int=128, min_snr::Real=5.)
	n_pix = length(y)
	for window_start in 1:Int(floor(window_width/10)):(n_pix - window_width)
		window_end = window_start + window_width
		mean_snr = sqrt(mean((y[window_start:window_end] .^2) ./ σ²[window_start:window_end]))
		if mean_snr > min_snr
			σ²[1:window_start] .= Inf # trim everything to left of window
			break
		end
	end
	for window_end in n_pix:-Int(floor(window_width/10)):window_width
		window_start = window_end - window_width
		mean_snr = sqrt(mean((y[window_start:window_end] .^2) ./ σ²[window_start:window_end]))
		if mean_snr > min_snr
			σ²[window_end:end] .= Inf # trim everything to right of window
			break
		end
	end
end
function mask_bad_edges!(tfd; kwargs...)
	for i in 1:size(tfd.log_λ_obs, 2)
		mask_bad_edges!(view(tfd.flux, :, i), view(tfd.var, :, i); kwargs...)
	end
end

function process!(tfd; kwargs...)
	mask_low_pixels!(tfd)
	mask_bad_edges!(tfd)
	continuum_normalize!(tfd; kwargs...)
	mask_high_pixels!(tfd)
end
