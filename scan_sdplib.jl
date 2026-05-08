using FileIO, SparseArrays, LinearAlgebra
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

files = readdir("ref/SDPLIB")
results = []

for f in files
    endswith(f, ".jld2") || continue
    name = replace(f, ".jld2" => "")
    try
        data = load("ref/SDPLIB/$f")
        A, b, C = sdplib_to_chordal(data["F"], data["c"])
        problem = Madeline.Problem(C, A, b)

        nc = length(Madeline.pointers(problem.cmp_to_col)) - 1
        n = data["n"]
        m = data["m"]

        # Compute connectivity %
        total_pairs = (m - problem.k) * (m - problem.k + 1) ÷ 2
        connected = 0
        mark = zeros(Int, m)
        for cj in problem.k+1:m
            for kj in Madeline.neighbors(problem.col_to_cmp, cj)
                for ci in Madeline.neighbors(problem.cmp_to_col, kj)
                    if ci >= cj && mark[ci] != cj
                        mark[ci] = cj
                        connected += 1
                    end
                end
            end
        end
        pct = total_pairs > 0 ? round(100*connected/total_pairs, digits=1) : 0.0

        push!(results, (name=name, n=n, m=m, k=problem.k, nc=nc, pct=pct))
    catch e
        println("Error on $name: $e")
    end
end

# Sort by number of components descending
sort!(results, by=x->x.nc, rev=true)

println("Top 20 problems by number of components:")
println("="^80)
for r in results[1:min(20, length(results))]
    println("$(rpad(r.name, 15)) n=$(lpad(r.n, 4)) m=$(lpad(r.m, 5)) k=$(lpad(r.k, 4)) components=$(lpad(r.nc, 4)) connected=$(lpad(r.pct, 5))%")
end

println("\n\nProblems with lowest connectivity %:")
println("="^80)
sort!(results, by=x->x.pct)
for r in results[1:min(20, length(results))]
    println("$(rpad(r.name, 15)) n=$(lpad(r.n, 4)) m=$(lpad(r.m, 5)) k=$(lpad(r.k, 4)) components=$(lpad(r.nc, 4)) connected=$(lpad(r.pct, 5))%")
end
