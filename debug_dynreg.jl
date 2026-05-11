using Pkg
Pkg.activate(@__DIR__)

using Madeline, JuMP, PowerModels
using CliqueTrees.Multifrontal: chol_impl!, DynamicRegularization
PowerModels.silence()

# Build problem
data = PowerModels.parse_file("ref/pglib-opf/pglib_opf_case57_ieee.m")
model = Model(Madeline.Optimizer)
# set_attribute(model, "iter_limit", 1)
set_attribute(model, "verbose", true)

pm = PowerModels.instantiate_model(data, PowerModels.SparseSDPWRMPowerModel, PowerModels.build_opf; jump_model=model)

# Try to get at the internals
backend = unsafe_backend(model)
println("Backend type: ", typeof(backend))

optimize!(model)

res = unsafe_backend(model).result
println("Status: ", Madeline.status(res))
