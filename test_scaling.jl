using FileIO, SparseArrays, LinearAlgebra, Test
using Madeline

function sdplib_to_chordal(F, c)
    m = length(c)
    n = size(F[1], 1)
    C = Hermitian(-F[1], :L)
    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]
    for j in 1:m
        Fj = F[j + 1]
        rows = rowvals(Fj)
        vals = nonzeros(Fj)
        for col in 1:n
            for k in nzrange(Fj, col)
                push!(I_idx, (col - 1) * n + rows[k])
                push!(J_idx, j)
                push!(V_val, -vals[k])
            end
        end
    end

    sparse(I_idx, J_idx, V_val, n^2, m), -c, C
end

name = "maxG11"
data = load("./ref/SDPLIB/$name.jld2")
obj_true = data["optVal"]

F = data["F"]
c = data["c"]

n = size(F[1], 1)
m = length(F) - 1

A, b, C = sdplib_to_chordal(F, c)
problem = Problem(C, A, b)

println("Problem: $name")
println("n = $n, m = $m")
println("Expected objective: $obj_true")
println()

# Test primal scaling
println("=== Primal Scaling ===")
settings_primal = Settings{Float64}(scaling=true, verbose=true)
solver_primal = Solver(problem)
result_primal = solve!(solver_primal; settings=settings_primal)
println("Status: $(status(result_primal))")
println("Objective: $(primal_objective(result_primal))")
println()

# Test dual scaling
println("=== Dual Scaling ===")
settings_dual = Settings{Float64}(scaling=false, verbose=true)
solver_dual = Solver(problem)
result_dual = solve!(solver_dual; settings=settings_dual)
println("Status: $(status(result_dual))")
println("Objective: $(primal_objective(result_dual))")
println()

# Verify both converged
@test status(result_primal) in OPTIMAL_STATES
@test status(result_dual) in OPTIMAL_STATES

# Verify both scalings give the same objective
@test isapprox(primal_objective(result_primal), primal_objective(result_dual), rtol=1e-6)

println("All tests passed!")
