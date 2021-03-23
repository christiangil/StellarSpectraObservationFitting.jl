function eval_regularization(train::TFOptimWorkspace, test::TFOptimWorkspace, loss_test::Function)
    train_TFModel!(train, 10)
    train_TFModel!(test, 5)
    return loss_test()
end

function fit_regularization_helper!(regs::Vector, reg_ind::Int, train::TFOptimWorkspace, test::TFOptimWorkspace, loss_test::Function; test_factor::Real=10)
    ℓs = zeros(2)
    reg_hold = [1, test_factor] .* regs[reg_ind]
    println("initial regularization eval")
    ℓs[1] = eval_regularization(train, test, loss_test)
    regs[reg_ind] = reg_hold[2]
    println("$(test_factor)x regularization eval")
    ℓs[2] = eval_regularization(train, test, loss_test)
    # need to try decreasing regularization
    if ℓs[2] > ℓs[1]
        while (ℓs[2] > ℓs[1]) && (1e-6 < reg_hold[1] < 1e12)
            println("trying a lower regularization")
            ℓs[2] = ℓs[1]
            reg_hold ./= test_factor
            regs[reg_ind] = reg_hold[1]
            ℓs[1] = eval_regularization(train, test, loss_test)
        end
        regs[reg_ind] = reg_hold[2]
    # need to try increasing regularization
    else
        while (ℓs[1] > ℓs[2]) && (1e-6 < reg_hold[2] < 1e12)
            println("trying a higher regularization")
            ℓs[1] = ℓs[2]
            reg_hold .*= test_factor
            regs[reg_ind] = reg_hold[2]
            ℓs[2] = eval_regularization(train, test, loss_test)
        end
        regs[reg_ind] = reg_hold[1]
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
    for i in 1:length(tfm.reg_tel)
        if i != 3
            test_factor = 10
        else
            test_factor = 1.2
        end

        println("before: reg_tel[$i]  = $(tfm.reg_tel[i])")
        fit_regularization_helper!(tfm.reg_tel, i, tf_workspace_train, tf_workspace_test, loss_test; test_factor=test_factor)
        println("after:  reg_tel[$i]  = $(tfm.reg_tel[i])")

        println("before: reg_star[$i] = $(tfm.reg_star[i])")
        fit_regularization_helper!(tfm.reg_star, i, tf_workspace_train, tf_workspace_test, loss_test; test_factor=test_factor)
        println("after:  reg_star[$i] = $(tfm.reg_star[i])")
    end
end
