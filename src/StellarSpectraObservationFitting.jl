module StellarSpectraObservationFitting

include("general_functions.jl")
include("DPCA_functions.jl")
include("model_functions.jl")
include("flatten.jl")
include("optimization_functions.jl")
include("regularization_functions.jl")
include("continuum_functions.jl")
include("model_selection_functions.jl")
include("mask_functions.jl")
include("Nabla_extension.jl")
include("TemporalGPs_extension.jl")

end # module
