## Setup
using Pkg
Pkg.activate("examples")
Pkg.instantiate()

using JLD2
import telfitting
# Pkg.status("telfitting")
# @time include("src/telfitting.jl")
tf = telfitting

## Setting up necessary variables and functions

@load "C:/Users/chris/OneDrive/Desktop/telfitting/tf_model_150k.jld2" tf_model n_obs tf_data

using StatsBase

n_obs_train = Int(round(0.75 * n_obs))
training_inds = sort(sample(1:n_obs, n_obs_train; replace=false))

function eval_regularization(train::tf.TFOptimWorkspace, test::tf.TFOptimWorkspace, loss_test::Function)
    for i in 1:8
        tf.train_TFModel!(train)
    end
    for i in 1:3
        tf.train_TFModel!(test)
    end
    return loss_test()
end

function fit_regularization_helper!(regs::Vector, reg_ind::Int, train::tf.TFOptimWorkspace, test::tf.TFOptimWorkspace, loss_test::Function; test_factor::Real=10)
    ℓs = zeros(2)
    reg_hold = [1, test_factor] .* regs[reg_ind]
    ℓs[1] = eval_regularization(train, test, loss_test)
    regs[reg_ind] = reg_hold[2]
    ℓs[2] = eval_regularization(train, test, loss_test)
    # need to try decreasing regularization
    if (ℓs[2] > ℓs[1]) && (1e-6 < reg_hold[1] < 1e12)
        while ℓs[2] > ℓs[1]
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

function fit_regularization!(tfm::tf.TFModel, tfd::tf.TFData, training_inds::AbstractVecOrMat)
    testing_inds = [i for i in 1:n_obs if !(i in training_inds)]
    println("creating workspaces")
    tf_workspace_train = tf.TFOptimWorkspace(tfm, tfd, training_inds)
    tf_workspace_test, loss_test = tf.TFOptimWorkspace(tfm, tfd, testing_inds; return_loss_f=true, only_s=true)
    println("starting regularization searches")
    for i in 1:1
    # for i in 1:length(tfm.reg_tel)
        if i != 3
            test_factor=10
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

fit_regularization!(tf_model, tf_data, training_inds)
