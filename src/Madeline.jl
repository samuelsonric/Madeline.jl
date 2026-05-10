module Madeline

using TimerOutputs
const TIMER = TimerOutput()

using LinearAlgebra
using LinearAlgebra: dot, norm, tr, axpy!, axpby!, ldiv!, lmul!, rmul!, Hermitian, I, mul!
using LinearAlgebra.BLAS: BlasInt
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange, nnz, sparse
using Printf
using CommonSolve: CommonSolve, init, solve, solve!, step!

using CliqueTrees: FBipartiteGraph, pointers, targets, neighbors, nv, etype, vertices, nov
using CliqueTrees.Multifrontal
using CliqueTrees.Multifrontal: HermOrSym, HermOrSymTri, HermOrSymSparse, HermTri, ChordalTriangular, FChordalTriangular, Permutation
using CliqueTrees.Multifrontal: FVector, FScalar, FMatrix, FPermutation, ChordalSymbolic
using CliqueTrees.Multifrontal: cholesky!, complete!, complete_dense!, uncholesky!, copyto!, similar, selaxpby!
using CliqueTrees.Multifrontal: fisher_impl!, fisherroot_impl!, complete_impl!, chol_impl!, unchol_impl!, selinv_impl!, div_impl!, mt_div_impl!, amari!, amari_impl!
using CliqueTrees.Multifrontal: potrf!, pstrf!, trsm!, trsv!, gemm!, gemv!, syrk!, symmtri!
using CliqueTrees.Multifrontal: fronts, diagblock, offdblock, ndz, ncl, nlz, nfr, half, eltypedegree, flatindex, getflatindex, setflatindex!, neighbors
using CliqueTrees.Multifrontal: logdet, cong, parent, triangular, symbolic, sympermute, symdot
using CliqueTrees.Multifrontal: oneto, ispositive, isnegative, one, zero, two, inv
using CliqueTrees.Multifrontal: DEFAULT_ELIMINATION_ALGORITHM

@enum Status begin
    CONTINUE
    OPTIMAL
    PRIMAL_INFEASIBLE
    DUAL_INFEASIBLE
    ILL_POSED
    NEAR_OPTIMAL
    NEAR_PRIMAL_INFEASIBLE
    NEAR_DUAL_INFEASIBLE
    NEAR_ILL_POSED
    SLOW_PROGRESS
    ITERATION_LIMIT
    NUMERICAL_FAILURE
end

const OPTIMAL_STATES = (OPTIMAL, NEAR_OPTIMAL)
const INFEASIBLE_STATES = (PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, NEAR_PRIMAL_INFEASIBLE, NEAR_DUAL_INFEASIBLE)
const ILL_POSED_STATES = (ILL_POSED, NEAR_ILL_POSED)

const ALPHA_SCHED = [
    0.9999, 0.999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8,
    0.7, 0.6, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005
]

const MAX_REF_STEPS = 5
const REF_MIN_IMPR = 0.5
const SPARSITY_THRESHOLD = 0.1

include("point.jl")
include("primal_dual_slack.jl")
include("problem.jl")
include("cholesky/cholesky.jl")
include("workspace.jl")
include("settings.jl")
include("state.jl")
include("utils.jl")
include("kkt.jl")
include("solver.jl")
include("result.jl")
include("equilibration/equilibration.jl")
include("MOI_wrapper.jl")

export Solver, Problem, Settings, Optimizer
export init, solve, solve!, step!
export equilibrate!, deequilibrate!
export Status
export OPTIMAL, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, ILL_POSED
export NEAR_OPTIMAL, NEAR_PRIMAL_INFEASIBLE, NEAR_DUAL_INFEASIBLE, NEAR_ILL_POSED
export SLOW_PROGRESS, ITERATION_LIMIT, NUMERICAL_FAILURE
export OPTIMAL_STATES, INFEASIBLE_STATES, ILL_POSED_STATES
export Result
export status, primal, dual, slack
export primal_objective, dual_objective, primal_inverse
export primal_residual, dual_residual, gap

end
