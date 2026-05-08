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

import Madeline
model = SOSModel(Madeline.Optimizer)
set_attribute(model, "verbose", false)
@variable(model, α)
@objective(model, Max, α)
@constraint(model, c, p >= α, domain = K, maxdegree = d)

# Access the backend optimizer
backend = JuMP.backend(model)

# Get the raw problem data
import MathOptInterface as MOI
cache = Madeline.OptimizerCache()
index_map = MOI.copy_to(cache, backend)

# Now manually extract data like MOI_wrapper does
psd_cones = Madeline._get_cones(cache, MOI.PositiveSemidefiniteConeTriangle)
nonneg_cones = Madeline._get_cones(cache, MOI.Nonnegatives)

println("PSD cones: ", length(psd_cones))
for (i, (ci, func, set)) in enumerate(psd_cones)
    println("  Cone $i: side_dimension = ", set.side_dimension, ", nvars = ", length(func.variables))
end

println("\nNonnegatives cones: ", length(nonneg_cones))
for (i, (ci, func, set)) in enumerate(nonneg_cones)
    println("  Cone $i: dimension = ", MOI.dimension(set), ", nvars = ", length(func.variables))
end

# Extract equality constraints
Ab = MOI.Utilities.constraints(cache.constraints, MOI.VectorAffineFunction{Float64}, MOI.Zeros)
A_moi = Ab.coefficients
b_moi = Ab.constants
m = length(b_moi)
println("\nEquality constraints: m = ", m)
println("max |b_moi| = ", maximum(abs, b_moi))
println("nnz(A_moi) = ", nnz(convert(SparseMatrixCSC, A_moi)))

# Objective
max_sense = MOI.get(cache, MOI.ObjectiveSense()) == MOI.MAX_SENSE
println("\nMax sense: ", max_sense)

obj_attr = MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
if obj_attr in MOI.get(cache, MOI.ListOfModelAttributesSet())
    obj = MOI.get(cache, obj_attr)
    println("Objective constant: ", obj.constant)
    println("Objective terms: ", length(obj.terms))
    for term in obj.terms
        println("  var $(term.variable.value): coef = $(term.coefficient)")
    end
end

# Now let's actually build the problem and inspect it
println("\n" * "="^60)
println("Building Madeline Problem")
println("="^60)

# Minimal reconstruction of the problem
n_vars = MOI.get(cache, MOI.NumberOfVariables())
println("Total MOI variables: ", n_vars)

# Build blocks
block_dims = Int[]
block_offsets = Int[]
varmap = fill((0, 0, 0), n_vars)

function trimap_inv(k)
    j = div(isqrt(8 * k - 7) + 1, 2)
    i = k - div(j * (j - 1), 2)
    return i, j
end

for (ci, func, set) in nonneg_cones
    d = MOI.dimension(set)
    blk = length(block_dims) + 1
    offset = isempty(block_offsets) ? 0 : block_offsets[end] + abs(block_dims[end])
    push!(block_dims, -d)
    push!(block_offsets, offset)
    for (k, vi) in enumerate(func.variables)
        varmap[vi.value] = (blk, k, k)
    end
end
for (ci, func, set) in psd_cones
    d = set.side_dimension
    blk = length(block_dims) + 1
    offset = isempty(block_offsets) ? 0 : block_offsets[end] + abs(block_dims[end])
    push!(block_dims, d)
    push!(block_offsets, offset)
    for (k, vi) in enumerate(func.variables)
        i, j = trimap_inv(k)
        varmap[vi.value] = (blk, i, j)
    end
end

n = block_offsets[end] + abs(block_dims[end])
println("Block-diagonal matrix dimension: n = ", n)
println("Block dims: ", block_dims)
println("Block offsets: ", block_offsets)

# Build C
C_I, C_J, C_V = Int[], Int[], Float64[]
if obj_attr in MOI.get(cache, MOI.ListOfModelAttributesSet())
    obj = MOI.get(cache, obj_attr)
    obj_sign = max_sense ? -1.0 : 1.0
    for term in obj.terms
        blk, i, j = varmap[term.variable.value]
        offset = block_offsets[blk]
        coef = obj_sign * term.coefficient
        if i != j
            coef /= 2
        end
        gi, gj = offset + i, offset + j
        if gi >= gj
            push!(C_I, gi); push!(C_J, gj); push!(C_V, coef)
        else
            push!(C_I, gj); push!(C_J, gi); push!(C_V, coef)
        end
    end
end
C = sparse(C_I, C_J, C_V, n, n)
println("\nC matrix:")
println("  nnz(C) = ", nnz(C))
println("  norm(C) = ", norm(C))
if nnz(C) <= 10
    for k in 1:nnz(C)
        i, j = C_I[k], C_J[k]
        println("  C[$i, $j] = $(C_V[k])")
    end
end

# Build A
A_I, A_J, A_V = Int[], Int[], Float64[]
A_sparse = convert(SparseMatrixCSC{Float64,Int}, A_moi)

for var_idx in axes(A_sparse, 2)
    for k in nzrange(A_sparse, var_idx)
        constraint_idx = rowvals(A_sparse)[k]
        coef = nonzeros(A_sparse)[k]
        blk, i, j = varmap[var_idx]
        offset = block_offsets[blk]
        gi, gj = offset + i, offset + j
        if i != j
            coef /= 2
        end
        flat_idx = if gi >= gj
            (gj - 1) * n + gi
        else
            (gi - 1) * n + gj
        end
        push!(A_I, flat_idx); push!(A_J, constraint_idx); push!(A_V, coef)
    end
end
A = sparse(A_I, A_J, A_V, n * n, m)
b = -b_moi

println("\nA matrix:")
println("  size(A) = ", size(A))
println("  nnz(A) = ", nnz(A))

println("\nb vector:")
println("  length(b) = ", length(b))
println("  norm(b) = ", norm(b))
println("  min(b) = ", minimum(b), ", max(b) = ", maximum(b))

# Check the problem structure
# The HSD embedding starts with x = I, z = I, y = solution of AAᵀy = Aᵀ(C - I)
# Compute the initial residuals

println("\n" * "="^60)
println("Checking initial point")
println("="^60)

# Initial primal X = I
X0 = Matrix{Float64}(I, n, n)
# Compute Aᵀ(X0) = trace of each constraint matrix
Ax0 = zeros(m)
for col in 1:m
    for k in nzrange(A, col)
        flat_idx = rowvals(A)[k]
        coef = nonzeros(A)[k]
        # flat_idx = (j-1)*n + i  where i >= j (lower triangular)
        j, rem = divrem(flat_idx - 1, n)
        i = rem + 1
        j = j + 1
        if i == j
            Ax0[col] += coef * X0[i, j]
        else
            Ax0[col] += 2 * coef * X0[i, j]  # symmetric contribution
        end
    end
end
println("Aᵀ(I) = ", Ax0[1:min(5, m)], "...")
println("norm(Aᵀ(I) - b) = ", norm(Ax0 - b))

# Compute ⟨C, I⟩
CI = sum(C[i, i] for i in 1:n)
println("⟨C, I⟩ = ", CI)

# What is the structure of the objective?
# We're maximizing α, which should correspond to one variable
# After negation for max sense, C should have a -1 somewhere for α
println("\nObjective interpretation:")
println("  With max_sense=true, we have min ⟨C, X⟩ = max ⟨-C, X⟩")
println("  So -⟨C, I⟩ = ", -CI)
