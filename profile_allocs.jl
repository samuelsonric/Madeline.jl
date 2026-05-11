using Pkg
Pkg.activate(@__DIR__)

using FileIO, SparseArrays, LinearAlgebra
using Madeline
using Profile

function load_truss6()
    data = load("./ref/SDPLIB/truss6.jld2")
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

problem = load_truss6()
settings = Settings{Float64}(; verbose=false)

# Warmup
solver = init(Madeline.Solver, problem)
solve!(solver; settings)

# Profile
println("=== Profiling allocations ===")
solver = init(Madeline.Solver, problem)
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 solve!(solver; settings)
results = Profile.Allocs.fetch()

vec_allocs = filter(a -> a.type == Vector{Int64}, results.allocs)
println("Found $(length(vec_allocs)) Vector{Int64} allocations")

# Look at stacktraces
println("\nSample stacktraces:")
for (i, alloc) in enumerate(vec_allocs[1:min(5, length(vec_allocs))])
    println("\n--- Allocation $i ---")
    for frame in alloc.stacktrace
        file = string(frame.file)
        if occursin("Madeline", file) || occursin("CliqueTrees", file)
            println("  $(basename(file)):$(frame.line) $(frame.func)")
        end
    end
end
