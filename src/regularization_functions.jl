function eval_regularization(train::TFOptimWorkspace, test::TFOptimWorkspace, loss_test::Function)
    train_TFModel!(train, 10)
    train_TFModel!(test, 5)
    return loss_test()
end

function fit_regularization_helper!(regs::Dict{Symbol, <:Real}, reg_key::Symbol, train::TFOptimWorkspace, test::TFOptimWorkspace, loss_test::Function; test_factor::Real=10)
    ℓs = zeros(2)
    reg_hold = [1, test_factor] .* regs[reg_key]
    println("initial regularization eval")
    ℓs[1] = eval_regularization(train, test, loss_test)
    regs[reg_key] = reg_hold[2]
    println("$(test_factor)x regularization eval")
    ℓs[2] = eval_regularization(train, test, loss_test)
    # need to try decreasing regularization
    if ℓs[2] > ℓs[1]
        while (ℓs[2] > ℓs[1]) && (1e-6 < reg_hold[1] < 1e12)
            println("trying a lower regularization")
            ℓs[2] = ℓs[1]
            reg_hold ./= test_factor
            regs[reg_key] = reg_hold[1]
            ℓs[1] = eval_regularization(train, test, loss_test)
        end
        regs[reg_key] = reg_hold[2]
    # need to try increasing regularization
    else
        while (ℓs[1] > ℓs[2]) && (1e-6 < reg_hold[2] < 1e12)
            println("trying a higher regularization")
            ℓs[1] = ℓs[2]
            reg_hold .*= test_factor
            regs[reg_key] = reg_hold[2]
            ℓs[2] = eval_regularization(train, test, loss_test)
        end
        regs[reg_key] = reg_hold[1]
    end
end


function check_for_valid_regularization(reg::Dict{Symbol, <:Real})
    for i in 1:length(reg.keys)
        try
            @assert reg.keys[i] in [:L1_M, :shared_M, :L2_M, :L2_μ, :L1_μ₊_factor, :L1_μ]
        catch y
            if !(typeof(y)==UndefRefError); error("The requested regularization isn't valid") end
        end
    end
end


function fit_regularization!(tfm::TFModel, tfd::TFData, training_inds::AbstractVecOrMat; use_telstar::Bool=true)
    n_obs = size(tfd.flux, 2)
    testing_inds = [i for i in 1:n_obs if !(i in training_inds)]
    println("creating workspaces")
    if use_telstar
        tf_workspace_train = TFWorkspaceTelStar(tfm, tfd, training_inds)
        tf_workspace_test, loss_test = TFWorkspaceTelStar(tfm, tfd, testing_inds; return_loss_f=true, only_s=true)
    else
        tf_workspace_train = TFWorkspace(tfm, tfd, training_inds)
        tf_workspace_test, loss_test = TFWorkspace(tfm, tfd, testing_inds; return_loss_f=true, only_s=true)
    end
    println("starting regularization searches")
    # for i in 3
    check_for_valid_regularization(tfm.reg_tel)
    check_for_valid_regularization(tfm.reg_star)
    for key in [:L1_M, :shared_M, :L2_M, :L2_μ, :L1_μ₊_factor, :L1_μ]
        key == :L1_μ₊_factor ? test_factor = 1.2 : test_factor = 10;

        if haskey(tfm.reg_tel, key)
            println("before: reg_tel[:$key]  = $(tfm.reg_tel[key])")
            fit_regularization_helper!(tfm.reg_tel, key, tf_workspace_train, tf_workspace_test, loss_test; test_factor=test_factor)
            println("after:  reg_tel[:$key]  = $(tfm.reg_tel[key])")
        end
        if haskey(tfm.reg_star, key)
            println("before: reg_star[:$key] = $(tfm.reg_star[key])")
            fit_regularization_helper!(tfm.reg_star, key, tf_workspace_train, tf_workspace_test, loss_test; test_factor=test_factor)
            println("after:  reg_star[:$key] = $(tfm.reg_star[key])")
        end
    end
end
