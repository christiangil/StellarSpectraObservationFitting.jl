## Setup
using Pkg
Pkg.activate("examples")
Pkg.instantiate()

using JLD2
import telfitting; tf = telfitting

## Setting up necessary variables and functions

@load "C:/Users/chris/OneDrive/Desktop/telfitting/tf_model_150k.jld2" tf_model n_obs tf_data

using StatsBase

n_obs_train = Int(round(0.75 * n_obs))
training_inds = sort(sample(1:n_obs, n_obs_train; replace=false))

use_telstar = true
tf.fit_regularization!(tf_model, tf_data, training_inds; use_telstar=use_telstar)
if use_telstar
    tf_workspace = tf.TFWorkspaceTelStar(tf_model, tf_data)
else
    tf_workspace = tf.TFWorkspace(tf_model, tf_data)
end
tf.train_TFModel!(tf_workspace, 10)
