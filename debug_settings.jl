using DynamicPolynomials
using SumOfSquares
using LinearAlgebra
using JuMP

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

import Madeline
model = SOSModel(Madeline.Optimizer)
println("Before setting attributes:")
println("  scaling = ", get_attribute(model, "scaling"))
println("  equilibration = ", get_attribute(model, "equilibration"))

set_attribute(model, "equilibration", false)
set_attribute(model, "scaling", false)

println("\nAfter setting attributes:")
println("  scaling = ", get_attribute(model, "scaling"))
println("  equilibration = ", get_attribute(model, "equilibration"))

@variable(model, α)
@objective(model, Max, α)
@constraint(model, c, p >= α, domain = K, maxdegree = d)
@time optimize!(model)
