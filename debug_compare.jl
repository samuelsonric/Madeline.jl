using DynamicPolynomials
using SumOfSquares
using LinearAlgebra

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

println("="^60)
println("Running Clarabel")
println("="^60)

using Clarabel
model = SOSModel(Clarabel.Optimizer)
@variable(model, α)
@objective(model, Max, α)
@constraint(model, c, p >= α, domain = K, maxdegree = d)
@time optimize!(model)
println("Clarabel objective: ", objective_value(model))
println()

println("="^60)
println("Running Madeline")
println("="^60)

import Madeline
model2 = SOSModel(Madeline.Optimizer)
@variable(model2, α)
@objective(model2, Max, α)
@constraint(model2, c, p >= α, domain = K, maxdegree = d)
@time optimize!(model2)
