using Pkg
Pkg.activate(@__DIR__)

using CliqueTrees.Multifrontal: DynamicRegularization, regularize

R = DynamicRegularization()
println("DynamicRegularization() fields:")
println("  delta   = ", R.delta)
println("  epsilon = ", R.epsilon)

# Test regularize with a small pivot
S = [1.0, 1.0, 1.0]
Djj_small = 1e-15
Djj_neg = -0.1

println("\nWith small pivot Djj = $Djj_small:")
println("  S[1] * Djj < epsilon? ", S[1] * Djj_small < R.epsilon)
println("  regularize returns: ", regularize(R, S, Djj_small, 1))

println("\nWith negative pivot Djj = $Djj_neg:")
println("  S[1] * Djj < epsilon? ", S[1] * Djj_neg < R.epsilon)
println("  regularize returns: ", regularize(R, S, Djj_neg, 1))
