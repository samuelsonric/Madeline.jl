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

# Build the problem directly
import MathOptInterface as MOI
import Madeline

model = SOSModel(Madeline.Optimizer)
set_attribute(model, "verbose", false)
set_attribute(model, "equilibration", false)
@variable(model, α)
@objective(model, Max, α)
@constraint(model, c_sos, p >= α, domain = K, maxdegree = d)

# Get the bridged model
optimize!(model)

# Access the raw data from the optimizer
backend = JuMP.backend(model)
# Navigate through the MOI layers
inner = backend.optimizer.model.optimizer
C, A, b = inner._C, inner._A, inner._b

println("Problem data:")
println("  n = ", size(C, 1))
println("  m = ", size(A, 2))
println("  nnz(C) = ", nnz(C))
println("  nnz(A) = ", nnz(A))

# Check the structure of C
println("\nC matrix (nonzeros):")
for j in 1:size(C, 2)
    for i in j:size(C, 1)
        if C[i, j] != 0
            println("  C[$i, $j] = ", C[i, j])
        end
    end
end

# Check norm of problem data
println("\nnorm(C) = ", norm(C))
println("norm(b) = ", norm(b))
println("max |Aij| = ", maximum(abs, nonzeros(A)))
println("min(b) = ", minimum(b), ", max(b) = ", maximum(b))

# Now let's manually construct the problem and step through
n = size(C, 1)
m = size(A, 2)

# The standard SDP form is:
#   (P)  min  ⟨C, X⟩
#        s.t. Aᵀ(X) = b
#             X ⪰ 0
#
#   (D)  max  ⟨b, y⟩
#        s.t. A(y) + Z = C
#             Z ⪰ 0

# Initial point: X = I, Z = I, y = 0, τ = κ = 1
X0 = Matrix{Float64}(I, n, n)
Z0 = Matrix{Float64}(I, n, n)
y0 = zeros(m)
τ0 = 1.0
κ0 = 1.0

# Compute primal objective ⟨C, X0⟩
pobj0 = dot(C, X0)
println("\nInitial primal objective: ", pobj0)

# Compute dual objective ⟨b, y0⟩
dobj0 = dot(b, y0)
println("Initial dual objective: ", dobj0)

# Compute primal residual: Aᵀ(X) - b
# We need to compute trace(Ai * X) for each constraint i
function apply_At(A, X, n)
    m = size(A, 2)
    result = zeros(m)
    for col in 1:m
        for k in nzrange(A, col)
            flat_idx = rowvals(A)[k]
            coef = nonzeros(A)[k]
            j, rem = divrem(flat_idx - 1, n)
            i = rem + 1
            j = j + 1
            if i == j
                result[col] += coef * X[i, j]
            else
                result[col] += 2 * coef * X[i, j]
            end
        end
    end
    return result
end

Ax0 = apply_At(A, X0, n)
pres0 = Ax0 - b
println("\nPrimal residual Aᵀ(X0) - b:")
println("  norm = ", norm(pres0))
println("  max |entry| = ", maximum(abs, pres0))

# Compute dual residual: A(y) + Z - C
function apply_A(A, y, n)
    result = zeros(n, n)
    for col in 1:size(A, 2)
        for k in nzrange(A, col)
            flat_idx = rowvals(A)[k]
            coef = nonzeros(A)[k]
            j, rem = divrem(flat_idx - 1, n)
            i = rem + 1
            j = j + 1
            result[i, j] += coef * y[col]
            if i != j
                result[j, i] += coef * y[col]
            end
        end
    end
    return result
end

Ay0 = apply_A(A, y0, n)
dres0 = Ay0 + Z0 - Matrix(C)
println("\nDual residual A(y0) + Z0 - C:")
println("  norm = ", norm(dres0))

# Gap
gap0 = pobj0 - dobj0
println("\nGap: ", gap0)

# The HSD embedding residual:
#   [    -b  Aᵀ ] [ y ]   [   ]
#   [ bᵀ    -cᵀ] [ τ ] = [ κ ]
#   [-A  c     ] [ x ]   [ z ]

# where cᵀ(x) = ⟨C, X⟩ and c = C (cost matrix)

# Residual in y-equation: -b*τ + Aᵀ(X) = 0
res_y = -b * τ0 + Ax0
println("\nHSD y-residual: -b*τ + Aᵀ(X)")
println("  norm = ", norm(res_y))

# Residual in τ-equation: bᵀy - ⟨C, X⟩ + κ = 0
res_tau = dot(b, y0) - dot(C, X0) + κ0
println("\nHSD τ-residual: bᵀy - ⟨C,X⟩ + κ = ", res_tau)

# Residual in x-equation: -A(y)*τ + C*τ - Z = 0
res_x = -Ay0 * τ0 + Matrix(C) * τ0 - Z0
println("\nHSD x-residual: -A(y)*τ + C*τ - Z")
println("  norm = ", norm(res_x))

# The normalized residuals
println("\n" * "="^60)
println("Comparing to Madeline iteration 0:")
println("="^60)

resy0 = max(1.0, norm(b))
resx0 = max(1.0, sqrt(sum(abs2(C[i,j]) * (i == j ? 1 : 2) for i in 1:n for j in 1:i if C[i,j] != 0)))

println("resy0 = ", resy0)
println("resx0 = ", resx0)

# Madeline shows at iter 0:
# gap = 74, pres = 1.86, dres = 7.32
# Let's check what those actually measure

# In update_state!:
#   pres = norm(rhs.dual) / resy0
#   dres = symnorm(rhs.slack.X) / resx0

# rhs comes from residual!(rhs, itr, problem)
# Looking at residual! in utils.jl:
#   res.dual ← α*[-b Aᵀ][τ;X] + β*res.dual  with α=-1, β=0

# So res.dual = -(-b*τ + Aᵀ(X)) = b*τ - Aᵀ(X)
# At initial point: b*1 - Aᵀ(I) = b - Aᵀ(I)
pres_madeline = norm(b - Ax0) / resy0
println("\nExpected pres (Madeline) = ", pres_madeline)

# And res.slack.X should be from: res.slack ← -A(y) + C*τ - Z (approximately)
# But there's cost! applied too...

# The gap in Madeline is:
# gap(state) = state.pobj - state.dobj
# And at iter 0: pobj = ⟨C, X0⟩/τ, dobj = ⟨b, y0⟩/τ
# But Madeline normalizes differently...

# Let's look at how Madeline computes state
# state.pobj = symdot(itr.primal.X, problem.C)
# state.dobj = dot(problem.b, itr.dual)

println("\nDirect computation:")
println("  pobj = ⟨C, I⟩ = ", pobj0)
println("  dobj = ⟨b, 0⟩ = ", dobj0)
println("  gap = pobj - dobj = ", gap0)
println()
println("Madeline shows gap = 74, which matches ", gap0)
