struct Result{UPLO, T, I}
    problem::Problem{T, I}
    state::State{UPLO, T, I}
end

function Result(solver::Solver)
    return Result(solver.problem, solver.curr)
end

function Base.show(io::IO, ::MIME"text/plain", result::Result)
    show(io, MIME"text/plain"(), result.state)
end

function status(result::Result)
    return result.state.status
end

#
# Variable accessors
#

function primal(result::Result{UPLO, T, I}) where {UPLO, T, I}
    τ = tau(result.state)
    itr = result.state.itr

    n = ncl(itr.primal.X)
    X = Matrix{T}(undef, n, n)

    complete_dense!(X, itr.primal.X)
    ldiv!(τ, X)

    return sympermute(X, result.problem.P.perm, 'L', 'L')
end

function primal_inverse(result::Result{UPLO, T, I}) where {UPLO, T, I}
    τ = tau(result.state)
    itr = result.state.itr

    L = copy(itr.primal.X)
    complete!(L)
    uncholesky!(L)
    lmul!(τ, L)

    return sympermute(L, result.problem.P.perm, 'L', 'L')
end

function dual(result::Result{UPLO, T, I}) where {UPLO, T, I}
    τ = tau(result.state)
    itr = result.state.itr

    m = length(itr.dual)
    y = Vector{T}(undef, m)

    mul!(y, result.problem.Q', itr.dual)
    ldiv!(τ, y)

    return y
end

function slack(result::Result{UPLO, T, I}) where {UPLO, T, I}
    τ = tau(result.state)
    itr = result.state.itr

    Z = sympermute(itr.slack.X, result.problem.P.perm, 'L', 'L')
    ldiv!(τ, Z)

    return Z
end

#
# Objective accessors
#

function primal_objective(result::Result)
    τ = tau(result.state)
    return result.state.pobj / τ
end

function dual_objective(result::Result)
    τ = tau(result.state)
    return result.state.dobj / τ
end

#
# Residual and gap accessors
#

function primal_residual(result::Result)
    τ = tau(result.state)
    return result.state.pres / τ
end

function dual_residual(result::Result)
    τ = tau(result.state)
    return result.state.dres / τ
end

function gap(result::Result)
    τ = tau(result.state)
    return gap(result.state) / τ^2
end

