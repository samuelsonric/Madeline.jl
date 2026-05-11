using Pkg
Pkg.activate(@__DIR__)

using FileIO, SparseArrays, LinearAlgebra
using Madeline

function sdplib_to_chordal(name::String)
    data = load("./ref/SDPLIB/$name.jld2")

    F = data["F"]
    c = data["c"]

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

    return Problem(C, sparse(I_idx, J_idx, V_val, n^2, m), -c)
end

for name in ["truss6", "truss7", "arch0"]
    println("\n=== $name ===")
    problem = sdplib_to_chordal(name)
    solver = init(Madeline.Solver, problem)
    result = solve!(solver)
    println("Status: ", status(result))
    println("Primal obj: ", primal_objective(result))
end
