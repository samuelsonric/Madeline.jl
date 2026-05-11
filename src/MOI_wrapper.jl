import MathOptInterface as MOI
using SparseArrays: sparse, SparseMatrixCSC, rowvals, nonzeros, nzrange, nnz
using LinearAlgebra: Hermitian

"""
    constraint_matrix(A, i, n) -> SparseMatrixCSC

Extract the i-th constraint matrix from A as an n×n sparse symmetric matrix.
A is stored as n²×m where column i contains the flattened lower-triangular
entries of the symmetric constraint matrix Aᵢ.
"""
function constraint_matrix(A::SparseMatrixCSC, i::Int, n::Int)
    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]

    for k in nzrange(A, i)
        flat_idx = rowvals(A)[k]
        val = nonzeros(A)[k]

        # Convert flat index to (row, col) using column-major: flat = (col-1)*n + row
        col, row_minus_1 = divrem(flat_idx - 1, n)
        row = row_minus_1 + 1
        col = col + 1

        push!(I_idx, row)
        push!(J_idx, col)
        push!(V_val, val)
        if row != col
            push!(I_idx, col)
            push!(J_idx, row)
            push!(V_val, val)
        end
    end

    return sparse(I_idx, J_idx, V_val, n, n)
end

# Supported conic sets (following SDPA pattern)
const _SupportedSets = Union{MOI.Nonnegatives,MOI.PositiveSemidefiniteConeTriangle}

# Cache structures
MOI.Utilities.@product_of_sets(ZeroCones, MOI.Zeros)

MOI.Utilities.@struct_of_constraints_by_set_types(
    ConstraintCache,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.Nonnegatives,
    MOI.Zeros,
)

const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    ConstraintCache{Float64}{
        MOI.Utilities.VectorOfConstraints{
            MOI.VectorOfVariables,
            MOI.PositiveSemidefiniteConeTriangle,
        },
        MOI.Utilities.VectorOfConstraints{
            MOI.VectorOfVariables,
            MOI.Nonnegatives,
        },
        MOI.Utilities.MatrixOfConstraints{
            Float64,
            MOI.Utilities.MutableSparseMatrixCSC{
                Float64,
                Int,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{Float64},
            ZeroCones{Float64},
        },
    },
}

mutable struct Optimizer <: MOI.AbstractOptimizer
    result::Union{Nothing,Result}
    equil::Union{Nothing,EquilibrationResult{Float64}}
    block_dims::Vector{Int}           # side dimension (negative for LP/diagonal blocks)
    block_offsets::Vector{Int}        # offset in block-diagonal matrix
    varmap::Vector{Tuple{Int,Int,Int}} # MOI variable index -> (block, i, j)
    cone_ci_to_blk::Dict{Int,Int}     # cone constraint index -> block number
    zeros_cones::Union{Nothing,ZeroCones{Float64}}
    max_sense::Bool
    objective_constant::Float64
    silent::Bool
    settings::Settings{Float64}
    _C::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    _A::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    _b::Union{Nothing,Vector{Float64}}

    function Optimizer()
        return new(
            nothing,
            nothing,
            Int[],
            Int[],
            Tuple{Int,Int,Int}[],
            Dict{Int,Int}(),
            nothing,
            false,
            0.0,
            false,
            Settings{Float64}(),
            nothing,
            nothing,
            nothing,
        )
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Madeline"

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

# RawOptimizerAttribute support for all settings
const _SUPPORTED_ATTRS = (
    "rel_opt", "abs_opt", "feas", "infeas", "tau_infeas",
    "illposed", "slow", "near_factor", "iter_limit", "prox_bound",
    "static_regularization", "dynamic_regularization_eps", "dynamic_regularization_delta",
    "scaling", "equilibration", "pivot", "verbose",
)

function MOI.supports(::Optimizer, attr::MOI.RawOptimizerAttribute)
    return attr.name in _SUPPORTED_ATTRS
end

function MOI.set(optimizer::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    name = attr.name
    if name == "rel_opt"
        optimizer.settings.rel_opt = Float64(value)
    elseif name == "abs_opt"
        optimizer.settings.abs_opt = Float64(value)
    elseif name == "feas"
        optimizer.settings.feas = Float64(value)
    elseif name == "infeas"
        optimizer.settings.infeas = Float64(value)
    elseif name == "tau_infeas"
        optimizer.settings.tau_infeas = Float64(value)
    elseif name == "illposed"
        optimizer.settings.illposed = Float64(value)
    elseif name == "slow"
        optimizer.settings.slow = Float64(value)
    elseif name == "near_factor"
        optimizer.settings.near_factor = Float64(value)
    elseif name == "iter_limit"
        optimizer.settings.iter_limit = Int(value)
    elseif name == "prox_bound"
        optimizer.settings.prox_bound = Float64(value)
    elseif name == "static_regularization"
        optimizer.settings.static_regularization = Float64(value)
    elseif name == "dynamic_regularization_eps"
        optimizer.settings.dynamic_regularization_eps = Float64(value)
    elseif name == "dynamic_regularization_delta"
        optimizer.settings.dynamic_regularization_delta = Float64(value)
    elseif name == "scaling"
        optimizer.settings.scaling = Bool(value)
    elseif name == "equilibration"
        optimizer.settings.equilibration = Bool(value)
    elseif name == "pivot"
        optimizer.settings.pivot = Bool(value)
    elseif name == "verbose"
        optimizer.settings.verbose = Bool(value)
    else
        throw(MOI.UnsupportedAttribute(attr))
    end
    return
end

function MOI.get(optimizer::Optimizer, attr::MOI.RawOptimizerAttribute)
    name = attr.name
    if name == "rel_opt"
        return optimizer.settings.rel_opt
    elseif name == "abs_opt"
        return optimizer.settings.abs_opt
    elseif name == "feas"
        return optimizer.settings.feas
    elseif name == "infeas"
        return optimizer.settings.infeas
    elseif name == "tau_infeas"
        return optimizer.settings.tau_infeas
    elseif name == "illposed"
        return optimizer.settings.illposed
    elseif name == "slow"
        return optimizer.settings.slow
    elseif name == "near_factor"
        return optimizer.settings.near_factor
    elseif name == "iter_limit"
        return optimizer.settings.iter_limit
    elseif name == "prox_bound"
        return optimizer.settings.prox_bound
    elseif name == "static_regularization"
        return optimizer.settings.static_regularization
    elseif name == "dynamic_regularization_eps"
        return optimizer.settings.dynamic_regularization_eps
    elseif name == "dynamic_regularization_delta"
        return optimizer.settings.dynamic_regularization_delta
    elseif name == "scaling"
        return optimizer.settings.scaling
    elseif name == "equilibration"
        return optimizer.settings.equilibration
    elseif name == "pivot"
        return optimizer.settings.pivot
    elseif name == "verbose"
        return optimizer.settings.verbose
    else
        throw(MOI.UnsupportedAttribute(attr))
    end
end

function MOI.is_empty(optimizer::Optimizer)
    return optimizer.result === nothing &&
           isempty(optimizer.block_dims) &&
           isempty(optimizer.varmap)
end

function MOI.empty!(optimizer::Optimizer)
    optimizer.result = nothing
    optimizer.equil = nothing
    empty!(optimizer.block_dims)
    empty!(optimizer.block_offsets)
    empty!(optimizer.varmap)
    empty!(optimizer.cone_ci_to_blk)
    optimizer.zeros_cones = nothing
    optimizer.max_sense = false
    optimizer.objective_constant = 0.0
    optimizer._C = nothing
    optimizer._A = nothing
    optimizer._b = nothing
    return
end

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    },
)
    return true
end

# Support both PSD and Nonnegatives (following SDPA pattern)
function MOI.supports_add_constrained_variables(
    ::Optimizer,
    ::Type{<:_SupportedSets},
)
    return true
end

# Free variables NOT supported
MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Reals}) = false

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{MOI.Zeros},
)
    return true
end

# Triangular dimension helpers
sympackedlen(d) = div(d * (d + 1), 2)

function trimap_inv(k)
    j = div(isqrt(8 * k - 7) + 1, 2)
    i = k - div(j * (j - 1), 2)
    return i, j
end

# Get cone data from cache
function _get_cones(src::OptimizerCache, ::Type{S}) where {S<:MOI.AbstractVectorSet}
    cache = MOI.Utilities.constraints(src.constraints, MOI.VectorOfVariables, S)
    indices = MOI.get(cache, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,S}())
    return [(ci, MOI.get(cache, MOI.ConstraintFunction(), ci),
             MOI.get(cache, MOI.ConstraintSet(), ci)) for ci in indices]
end

# Add a PSD block (following SDPA _new_block pattern)
function _new_block!(dest::Optimizer, ci, func, set::MOI.PositiveSemidefiniteConeTriangle)
    d = set.side_dimension
    blk = length(dest.block_dims) + 1
    # Compute offset before adding to block_dims
    offset = isempty(dest.block_offsets) ? 0 : dest.block_offsets[end] + abs(dest.block_dims[end])
    push!(dest.block_dims, d)  # positive = SDP block
    push!(dest.block_offsets, offset)
    dest.cone_ci_to_blk[ci.value] = blk
    for (k, vi) in enumerate(func.variables)
        i, j = trimap_inv(k)
        dest.varmap[vi.value] = (blk, i, j)
    end
    return d
end

# Add a Nonnegatives block (diagonal block, following SDPA pattern)
function _new_block!(dest::Optimizer, ci, func, set::MOI.Nonnegatives)
    d = MOI.dimension(set)
    blk = length(dest.block_dims) + 1
    # Compute offset before adding to block_dims
    offset = isempty(dest.block_offsets) ? 0 : dest.block_offsets[end] + abs(dest.block_dims[end])
    push!(dest.block_dims, -d)  # negative = LP/diagonal block
    push!(dest.block_offsets, offset)
    dest.cone_ci_to_blk[ci.value] = blk
    for (k, vi) in enumerate(func.variables)
        dest.varmap[vi.value] = (blk, k, k)  # diagonal entries only
    end
    return d
end

function _optimize!(dest::Optimizer, src::OptimizerCache)
    MOI.empty!(dest)

    psd_cones = _get_cones(src, MOI.PositiveSemidefiniteConeTriangle)
    nonneg_cones = _get_cones(src, MOI.Nonnegatives)

    if isempty(psd_cones) && isempty(nonneg_cones)
        error("Madeline requires at least one PSD or Nonnegatives constraint")
    end

    # Initialize varmap
    n_vars = MOI.get(src, MOI.NumberOfVariables())
    resize!(dest.varmap, n_vars)
    fill!(dest.varmap, (0, 0, 0))

    # Build blocks (process Nonnegatives first like SDPA does)
    for (ci, func, set) in nonneg_cones
        _new_block!(dest, ci, func, set)
    end
    for (ci, func, set) in psd_cones
        _new_block!(dest, ci, func, set)
    end

    # Total dimension
    n = dest.block_offsets[end] + abs(dest.block_dims[end])

    # Check all variables mapped
    mapped_count = count(x -> x != (0, 0, 0), dest.varmap)
    if mapped_count != n_vars
        error("Madeline requires all variables in PSD or Nonnegatives cones. " *
              "Found $n_vars variables but only $mapped_count mapped. " *
              "Use MOI.Bridges.full_bridge_optimizer to bridge free variables.")
    end

    # Extract equality constraints
    Ab = MOI.Utilities.constraints(src.constraints, MOI.VectorAffineFunction{Float64}, MOI.Zeros)
    A_moi = Ab.coefficients
    b_moi = Ab.constants
    m = length(b_moi)

    if m == 0
        error("Madeline requires at least one equality constraint")
    end

    # Build objective
    dest.max_sense = MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    obj_attr = MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
    dest.objective_constant = if obj_attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        MOI.constant(MOI.get(src, obj_attr))
    else
        0.0
    end

    # Build C matrix
    C_I, C_J, C_V = Int[], Int[], Float64[]
    if obj_attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        obj = MOI.get(src, obj_attr)
        obj_sign = dest.max_sense ? -1.0 : 1.0
        for term in obj.terms
            blk, i, j = dest.varmap[term.variable.value]
            offset = dest.block_offsets[blk]
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

    # Build A matrix
    A_I, A_J, A_V = Int[], Int[], Float64[]
    A_sparse = convert(SparseMatrixCSC{Float64,Int}, A_moi)

    for var_idx in axes(A_sparse, 2)
        for k in nzrange(A_sparse, var_idx)
            constraint_idx = rowvals(A_sparse)[k]
            coef = nonzeros(A_sparse)[k]
            blk, i, j = dest.varmap[var_idx]
            offset = dest.block_offsets[blk]
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

    dest.zeros_cones = deepcopy(Ab.sets)
    dest._C, dest._A, dest._b = C, A, copy(b)

    # Solve
    problem = Problem(Hermitian(C, :L), A, b)
    if dest.settings.equilibration
        dest.equil = equilibrate!(problem)
    else
        dest.equil = EquilibrationResult{Float64}(m)
    end
    dest.settings.verbose = !dest.silent
    dest.result = solve(problem; settings=dest.settings)
    deequilibrate!(dest.result, dest.equil)

    return
end

function MOI.optimize!(dest::Optimizer, src::OptimizerCache)
    _optimize!(dest, src)
    return MOI.Utilities.identity_index_map(src), false
end

function MOI.optimize!(dest::Optimizer, src::MOI.Utilities.UniversalFallback{OptimizerCache})
    MOI.Utilities.throw_unsupported(src)
    return MOI.optimize!(dest, src.model)
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    _optimize!(dest, cache)
    return index_map, false
end

MOI.supports_incremental_interface(::Optimizer) = false

# Status mappings
function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    if optimizer.result === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    s = status(optimizer.result)
    if s == OPTIMAL
        return MOI.OPTIMAL
    elseif s == NEAR_OPTIMAL
        return MOI.ALMOST_OPTIMAL
    elseif s == PRIMAL_INFEASIBLE
        return MOI.INFEASIBLE
    elseif s == NEAR_PRIMAL_INFEASIBLE
        return MOI.ALMOST_INFEASIBLE
    elseif s == DUAL_INFEASIBLE
        return MOI.DUAL_INFEASIBLE
    elseif s == NEAR_DUAL_INFEASIBLE
        return MOI.ALMOST_DUAL_INFEASIBLE
    elseif s == ILL_POSED || s == NEAR_ILL_POSED
        return MOI.INFEASIBLE_OR_UNBOUNDED
    elseif s == SLOW_PROGRESS
        return MOI.SLOW_PROGRESS
    elseif s == ITERATION_LIMIT
        return MOI.ITERATION_LIMIT
    elseif s == NUMERICAL_FAILURE
        return MOI.NUMERICAL_ERROR
    else
        return MOI.OTHER_ERROR
    end
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return optimizer.result === nothing ? "optimize! not called" : string(status(optimizer.result))
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    s = status(optimizer.result)
    if s in OPTIMAL_STATES
        return MOI.FEASIBLE_POINT
    elseif s == DUAL_INFEASIBLE || s == NEAR_DUAL_INFEASIBLE
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif s == PRIMAL_INFEASIBLE || s == NEAR_PRIMAL_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    s = status(optimizer.result)
    if s in OPTIMAL_STATES
        return MOI.FEASIBLE_POINT
    elseif s == PRIMAL_INFEASIBLE || s == NEAR_PRIMAL_INFEASIBLE
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif s == DUAL_INFEASIBLE || s == NEAR_DUAL_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = optimizer.result === nothing ? 0 : 1

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    val = primal_objective(optimizer.result)
    return (optimizer.max_sense ? -val : val) + optimizer.objective_constant
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    val = dual_objective(optimizer.result)
    return (optimizer.max_sense ? -val : val) + optimizer.objective_constant
end

# Variable primal
function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    X = primal(optimizer.result)
    blk, i, j = optimizer.varmap[vi.value]
    offset = optimizer.block_offsets[blk]
    gi, gj = offset + i, offset + j
    return gi >= gj ? X[gi, gj] : X[gj, gi]
end

# Constraint primal for cones (following SDPA _vectorize_block pattern)
function _vectorize_block(X, blk, offset, ::Type{MOI.Nonnegatives}, d)
    return [X[offset + k, offset + k] for k in 1:d]
end

function _vectorize_block(X, blk, offset, ::Type{MOI.PositiveSemidefiniteConeTriangle}, d)
    result = Vector{Float64}(undef, sympackedlen(d))
    k = 0
    for j in 1:d, i in 1:j
        k += 1
        result[k] = X[offset + i, offset + j]
    end
    return result
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables,S},
) where {S<:_SupportedSets}
    MOI.check_result_index_bounds(optimizer, attr)
    blk = optimizer.cone_ci_to_blk[ci.value]
    d = abs(optimizer.block_dims[blk])
    offset = optimizer.block_offsets[blk]
    return _vectorize_block(primal(optimizer.result), blk, offset, S, d)
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},MOI.Zeros},
)
    MOI.check_result_index_bounds(optimizer, attr)
    return zeros(length(MOI.Utilities.rows(optimizer.zeros_cones, ci)))
end

# Constraint dual
function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables,S},
) where {S<:_SupportedSets}
    MOI.check_result_index_bounds(optimizer, attr)
    blk = optimizer.cone_ci_to_blk[ci.value]
    d = abs(optimizer.block_dims[blk])
    offset = optimizer.block_offsets[blk]
    return _vectorize_block(slack(optimizer.result), blk, offset, S, d)
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},MOI.Zeros},
)
    MOI.check_result_index_bounds(optimizer, attr)
    y = dual(optimizer.result)
    rows = MOI.Utilities.rows(optimizer.zeros_cones, ci)
    return -y[rows]
end
