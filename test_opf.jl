using Pkg
Pkg.activate(@__DIR__)

using SparseArrays, LinearAlgebra, JuMP, Madeline, PowerModels

# Silence PowerModels output
PowerModels.silence()

# Load a test case (MATPOWER .m file)
filepath = "ref/pglib-opf/pglib_opf_case57_ieee.m"
# filepath = "ref/pglib-opf/pglib_opf_case3_lmbd.m"
data = PowerModels.parse_file(filepath)

# Build SDP relaxation
model = Model(Madeline.Optimizer)
set_attribute(model, "iter_limit", 5)

pm = PowerModels.instantiate_model(
    data,
    PowerModels.SparseSDPWRMPowerModel,
    PowerModels.build_opf;
    jump_model = model
)

println("=== Solving $filepath ===")
@time optimize!(model)

res = unsafe_backend(model).result
println("Status: ", Madeline.status(res))
