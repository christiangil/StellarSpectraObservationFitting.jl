using Pkg
Pkg.activate("10700")
Pkg.instantiate()

using JLD2
import telfitting; tf = telfitting

## Setting up necessary variables and functions

@load "C:/Users/chris/OneDrive/Desktop/telfitting/10700.jld2" tf_model n_obs tf_data rvs_naive rvs_notel

using StatsBase

n_obs_train = Int(round(0.75 * n_obs))
training_inds = sort(sample(1:n_obs, n_obs_train; replace=false))
use_telstar = true

tf.fit_regularization!(tf_model, tf_data, training_inds; use_telstar=use_telstar)
println(tf_model.reg_tel)
println(tf_model.reg_star)
if use_telstar
    tf_workspace = tf.TFWorkspaceTelStar(tf_model, tf_data)
else
    tf_workspace = tf.TFWorkspace(tf_model, tf_data)
end
tf.train_TFModel!(tf_workspace, 10)
