#setup.jl
using Pkg; using Pkg.API
Pkg.add("NPZ")
Pkg.add("UnitfulAstro")
Pkg.add("Unitful")
Pkg.add("HDF5")
Pkg.add("Distributions")
Pkg.add("Stheno")
Pkg.add("TemporalGPs")
Pkg.add("Zygote")
Pkg.add("Plots")
Pkg.add("Optim")
Pkg.add("LineSearches")
Pkg.add("Flux")
# Pkg.add("JLD2")
Pkg.API.precompile()

Pkg.update()
Pkg.activate(".")
Pkg.instantiate()
Pkg.update()
Pkg.API.precompile()

Pkg.status()
