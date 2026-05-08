using DynamicPolynomials
using SumOfSquares
using LinearAlgebra
using SparseArrays

c = [42, 44, 45, 47, 47.5]
Q = 100I

@polyvar x[1:5]
p = c'x - x' * Q * x / 2
K = @set x[1] >= 0 && x[1] <= 1 &&
         x[2] >= 0 && x[2] <= 1 &&
         x[3] >= 0 && x[3] <= 1 &&
         x[4] >= 0 && x[4] <= 1 &&
         x[5] >= 0 && x[5] <= 1 &&
         10x[1] + 12x[2] + 11x[3] + 7x[4] + 4x[5] <= 40

d = 3

# Build model using SumOfSquares but extract data and solve directly
import MathOptInterface as MOI
import Madeline

# First build the model to get the problem data
model = SOSModel(Madeline.Optimizer)
set_attribute(model, "verbose", false)
@variable(model, α)
@objective(model, Max, α)
@constraint(model, c_sos, p >= α, domain = K, maxdegree = d)
optimize!(model)

# Get the raw data
backend = JuMP.backend(model)
inner = backend.optimizer.model.optimizer
C, A, b = inner._C, inner._A, inner._b

println("Problem dimensions:")
println("  n = ", size(C, 1))
println("  m = ", size(A, 2))

# Now solve directly with Madeline API
problem = Madeline.Problem(Hermitian(C, :L), A, b)

println("\nSolving with primal scaling (default):")
settings_primal = Madeline.Settings{Float64}(
    scaling = true,
    verbose = true,
)
result_primal = Madeline.solve(problem; settings=settings_primal)

println("\n\n")
println("="^60)
println("Solving with dual scaling:")
println("="^60)

# Need to create fresh problem since solve! modifies it
problem2 = Madeline.Problem(Hermitian(copy(C), :L), copy(A), copy(b))
settings_dual = Madeline.Settings{Float64}(
    scaling = false,
    verbose = true,
)
result_dual = Madeline.solve(problem2; settings=settings_dual)
