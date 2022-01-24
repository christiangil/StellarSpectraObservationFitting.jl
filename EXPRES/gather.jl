## Importing packages
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting

## Setting up necessary variables

input_ind = SSOF.parse_args(1, Int, 0)
only_excalibur = SSOF.parse_args(2, Bool, false)

stars = ["10700", "26965", "34411"]
input_ind == 0 ? star_inds = (1:3) : star_inds = input_ind
if only_excalibur
    # excals = Bool.(zeros(85))
    # for order in 1:85
    #     @load expres_save_path * star * "/$(order)/data.jld2" used_excal
    #     excals[order] = used_excal
    # end
    # findfirst(excals):findlast(excals)
    orders_list = [42:77, 40:77, 38:77]
else
    orders_list = [1:85, 1:85, 1:85]
end
include("data_locs.jl")  # defines expres_data_path and expres_save_path
# prep_str = "noreg_"
prep_str = ""

function retrieve(order::Int, star::String)
    @load expres_save_path*star*"/$(order)/$(prep_str)results.jld2" model rv_errors
    rvs_notel_opt = SSOF.rvs(model)
    return rvs_notel_opt, rv_errors
end

function retrieve_md(order::Int, star::String)
    @load expres_save_path*star*"/$(order)/$(prep_str)model_decision.jld2" comp_ls ℓ aics bics ks test_n_comp_tel test_n_comp_star
    ans_aic = argmin(aics)
    ans_bic = argmin(bics)
    n_comps = [test_n_comp_tel[ans_aic[1]], test_n_comp_star[ans_aic[2]]]
    n_comps_bic = [test_n_comp_tel[ans_bic[1]], test_n_comp_star[ans_bic[2]]]
    return n_comps, n_comps_bic, ans_aic==ans_bic
end

function retrieve_lcrvs(order::Int, star::String)
    @load expres_save_path*star*"/$(order)/$(prep_str)low_comp_rvs.jld2" rvs rvs_σ test_n_comp_tel test_n_comp_star
    return rvs, rvs_σ
end

reg_keys = SSOF._key_list[1:end-1]
function retrieve_reg(order::Int, star::String)
    @load expres_save_path*star*"/$(order)/$(prep_str)results.jld2" model
    return [model.reg_tel[k] for k in reg_keys], [model.reg_star[k] for k in reg_keys]
end

for star_ind in star_inds
    star = stars[star_ind]
    orders = orders_list[star_ind]
    n_ord = length(orders)
    @load expres_save_path*star*"/$(orders[1])/data.jld2" n_obs times_nu airmasses

    rvs = zeros(n_ord,  n_obs)
    rvs_σ = zeros(n_ord, n_obs)
    for i in 1:n_ord
        try
            rvs[i, :], rvs_σ[i, :] = retrieve(orders[i], star)
        catch
            rvs_σ[i, :] .= Inf
            println("order $(orders[i]) is missing")
        end
    end
    @save "expres_$(prep_str)$(star)_rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord

    @load expres_save_path*star*"/$(orders[1])/data.jld2" n_obs times_nu airmasses
    n_comps = zeros(Int, n_ord, 2)
    n_comps_bic = zeros(Int, n_ord, 2)
    robust = zeros(Bool, n_ord)
    for i in 1:n_ord
        try
            n_comps[i, :], n_comps_bic[i, :], robust[i] = retrieve_md(orders[i], star)
        catch
            n_comps[i, :] .= -1
            println("order $(orders[i]) is missing")
        end
    end
    @save "expres_$(prep_str)$(star)_md.jld2" n_comps n_comps_bic robust

    # rvs = zeros(n_ord, 3, 3, n_obs)
    # rvs_σ = zeros(n_ord, 3, 3, n_obs)
    # for i in 1:n_ord
    #     try
    #         rvs[i, :, :, :], rvs_σ[i, :, :, :] = retrieve_lcrvs(orders[i], star)
    #     catch
    #         rvs_σ[i, :, :, :] .= Inf
    #         println("order $(orders[i]) is missing")
    #     end
    # end
    # @save "expres_$(prep_str)$(star)_lcrvs.jld2" rvs rvs_σ

    reg_tels = zeros(n_ord, length(reg_keys))
    reg_stars = zeros(n_ord, length(reg_keys))
    for i in 1:n_ord
        try
            reg_tels[i, :], reg_stars[i, :] = retrieve_reg(orders[i], star)
        catch
            println("order $(orders[i]) is missing")
        end
    end
    @save "expres_$(prep_str)$(star)_regs.jld2" reg_tels reg_stars
end
