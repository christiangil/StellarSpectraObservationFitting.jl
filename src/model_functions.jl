function create_λ_template(log_λ, resolution)
    log_min_wav, log_max_wav = [minimum(log_λ), maximum(log_λ)]
    len = Int(ceil((exp(log_max_wav) - exp(log_min_wav)) * resolution / exp((log_max_wav + log_min_wav)/2)))
    log_Δλ = (log(log_max_wav) - log(log_min_wav)) / len
    len += 2
    log_λ_template = range(log_min_wav - log_Δλ; length = len,  stop = log_max_wav + log_Δλ)
    λ_template = exp.(log_λ_template)
    return len, log_λ_template, λ_template
end

L1(thing) = sum(abs.(thing))
L2(thing) = sum(thing .* thing)

linear_model(θ) = (θ[1] * θ[2]) .+ θ[3]  # M * s + template
function model_prior(θ, coeffs::Vector{<:Real})
    template_mod = θ[3] .- 1
    return (coeffs[1] * sum(template_mod[template_mod.>0])) +
    (coeffs[2] * L1(template_mod)) +
    (coeffs[3] * L2(template_mod)) +
    (coeffs[4] * L1(θ[1])) +
    (coeffs[5] * L2(θ[1])) +
    L1(θ[2])
end

tel_model(θ) = linear_model(θ)
