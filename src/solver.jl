# Madeline.jl solves the primal-dual pair of semidefinite
# programs.
#
#     (P)  minimize   ⟨x, c⟩
#          subect to  Ax = b
#                      x ≥ 0
#
#     (D)  maximize   ⟨b, y⟩
#          subject to  Aᵀy + z = c
#                      z ≥ 0 
#
# By analysing the sparsity of A and c, we can find a low-
# dimensional subspace V such that
#
#      c - Aᵀy ∈ V
#
# for all vectors y*. This yields a new pair of conic programs
#
#     (P)  minimize    ⟨x, c⟩
#          subject to  Ax = b
#                      x ∈ K
#
#     (D)  maximize    ⟨b, y⟩
#          subject to  Aᵀy + z = c
#                      z ∈ K*
#
# where V⁰ is the annihilator of V and
#
#      K  := { x  | x  ≥ 0 } / V⁰
#      K* := { z* | z* ≥ 0 } ∩ V .
#
# We then construct a homogeneous self-dual embedding:
#
#          find        (y, τ, x, κ, z)
#
#          subject to  [    -b  A  ] [ y ]   [   ]
#                      [ bᵀ    -cᵀ ] [ τ ] = [ κ ]
#                      [-Aᵀ  c     ] [ x ]   [ z ]
#
#                      [ τ ] ∈ ℝ⁺ × K
#                      [ x ]
#
#                      [ κ ] ∈ ℝ⁺ × K*
#                      [ z ]
#
# and solve it using Skajaa and Ye's nonsymmetric interior
# point algorithm.

struct Solver{UPLO, T, J, Chol <: AbstractCholesky{T}}
    curr::State{T}
    prev::State{T}
    space::Workspace{T, J}
    cache::KKT{T, Chol}
    itr::PrimalDualSlack{UPLO, T, J}
    pd1::PrimalDualSlack{UPLO, T, J}
    pd2::PrimalDualSlack{UPLO, T, J}
    cd1::PrimalDualSlack{UPLO, T, J}
    cd2::PrimalDualSlack{UPLO, T, J}
    rhs::PrimalDualSlack{UPLO, T, J}
    wrk::PrimalDualSlack{UPLO, T, J}
    res::PrimalDualSlack{UPLO, T, J}
    q::Primal{UPLO, T, J}
    L::FChordalTriangular{:N, UPLO, T, J}
    problem::Problem{T, J}
    resy0::T
    resx0::T
end

function Solver(problem::Problem{T, J}) where {T<:Real, J<:Integer}
    G = problem.G
    C = problem.C
    A = problem.A
    b = problem.b
    S = problem.S
    P = problem.P
    Q = problem.Q

    n = size(C, 1)
    m = size(A, 2)

    cache = KKT{T}(problem)

    itr = PrimalDualSlack{:L, T}(m, S, G)
    initialize!(itr, cache, problem)

    q = Primal{:L, T}(S)
    L = similar(itr.primal.X)

    rhs = PrimalDualSlack{:L, T}(m, S, G)
    pd1 = PrimalDualSlack{:L, T}(m, S, G)
    pd2 = PrimalDualSlack{:L, T}(m, S, G)
    cd1 = PrimalDualSlack{:L, T}(m, S, G)
    cd2 = PrimalDualSlack{:L, T}(m, S, G)
    wrk = PrimalDualSlack{:L, T}(m, S, G)
    res = PrimalDualSlack{:L, T}(m, S, G)

    space = Workspace{T}(S, size(cache.U, 2))

    curr = State{T}(n)
    prev = State{T}(n)

    resy0 = max(one(T), norm(b))
    resx0 = max(one(T), symnorm(C))

    return Solver(
        curr, prev, space, cache,
        itr, pd1, pd2, cd1, cd2, rhs, wrk, res,
        q, L, problem,
        resy0, resx0)
end

function show_solver(io::IO, solver::Solver, indent::Int)
    pad = " "^indent
    println(io, pad, "problem:")
    show_problem(io, solver.problem, indent + 2)
    println(io)
    println(io)
    println(io, pad, "state:")
    show_state(io, solver.curr, indent + 2)
    return
end

function Base.show(io::IO, ::MIME"text/plain", solver::Solver{UPLO, T, J}) where {UPLO, T, J}
    println(io, "Solver{:$UPLO, $T, $J}:")
    show_solver(io, solver, 2)
end

function update_state!(
        state::State{T},
        prev::State{T},
        itr::PrimalDualSlack{UPLO, T, J},
        rhs::PrimalDualSlack{UPLO, T, J},
        res::PrimalDualSlack{UPLO, T, J},
        problem::Problem{T, J},
        settings::Settings{T},
        resy0::T,
        resx0::T,
    ) where {UPLO, T, J}
    n = state.ncol

    state.τ = itr.primal.τ
    state.κ = itr.slack.τ
    state.pobj = symdot(itr.primal.X, problem.C)
    state.dobj = dot(problem.b, itr.dual)
    state.μ = dot(itr.primal, itr.slack) / (n + 1)

    residual!(rhs, itr, problem)
    state.pres    = norm(rhs.dual)       / resy0
    state.dres    = symnorm(rhs.slack.X) / resx0
    state.pinf = typemax(T)
    state.dinf = typemax(T)

    if ispositive(state.dobj)
        copyto_sparse!(res.slack.X, itr.slack.X)
        apply_constraint!(problem.A, problem.indices_slack, res.slack.X, itr.dual, one(T), one(T), Val(false))
        state.pinf = symnorm(res.slack.X) / resx0 / state.dobj
    end

    if isnegative(state.pobj)
        apply_constraint!(problem.A, problem.indices_primal, itr.primal.X, res.dual, one(T), zero(T), Val(true))
        state.dinf = norm(res.dual) / resy0 / (-state.pobj)
    end

    state.status = check_termination(state, settings)

    if state.status == CONTINUE
        check_slow_progress!(state, prev, settings)
    end

    return state
end

function check_slow_progress!(state::State{T}, prev::State{T}, settings::Settings{T}) where {T}
    Δ = zero(T)

    p = prev.pres  / prev.τ
    q = state.pres / state.τ
    Δ = max(Δ, (p - q) / (abs(p) + eps(T)))

    p = prev.dres  / prev.τ
    q = state.dres / state.τ
    Δ = max(Δ, (p - q) / (abs(p) + eps(T)))

    p = gap(prev)  / prev.τ^2
    q = gap(state) / state.τ^2
    Δ = max(Δ, (p - q) / (abs(p) + eps(T)))

    if state.nitr > 0 && Δ < settings.slow
        state.nslw += 1
    else
        state.nslw = 0
    end

    if state.nslw >= 3
        state.status = SLOW_PROGRESS
    end

    return state
end

function proximity!(
        space::Workspace{T, J},
        res::PrimalDualSlack{:L, T, J},
        itr::PrimalDualSlack{:L, T, J},
        q::Primal{:L, T, J},
        L::ChordalTriangular{:N, :L, T, J},
        μ::T,
        scaling::Val{SCALE},
    ) where {T, J, SCALE}
    w = res.primal

    if SCALE
        s = itr.slack
        x = itr.primal
    else
        s = itr.primal
        x = q
    end

    copyto!(w, s)
    axpy!(-μ, q, w)
    hessianroot!(space, L, x, w, flip(scaling))

    return norm(w) / μ
end

function initialize!(
        itr::PrimalDualSlack{UPLO, T, J},
        cache::KKT{T},
        problem::Problem{T, J},
    ) where {UPLO, T, J}
    build_gram!(cache, problem.A)

    copyto!(itr.primal.X, problem.C)
    axpy!(-one(T), I, itr.primal.X)
    apply_constraint!(problem.A, problem.indices_primal, itr.primal.X, itr.dual, one(T), zero(T), Val(true))

    ldiv_fwd!(cache.chol, itr.dual)
    ldiv_bwd!(cache.chol, itr.dual)

    itr.primal.τ = one(T)
    itr.slack.τ = one(T)

    copyto!(itr.primal.X, I)
    copyto!(itr.slack.X, I)

    return itr
end

function corrector_rhs!(
        rhs::PrimalDualSlack{UPLO, T, J},
        itr::PrimalDualSlack{UPLO, T, J},
        q::Primal{UPLO, T, J},
        μ::T,
        ::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    if SCALE
        z = itr.slack
    else
        z = itr.primal
    end

    fill!(rhs.slack, zero(T))
    fill!(rhs.dual, zero(T))

    copyto!(rhs.primal, z)
    axpby!(μ, q, -one(T), rhs.primal)
    return
end

function prediction_toa_rhs!(
        space::Workspace{T, J},
        wrk::PrimalDualSlack{UPLO, T, J},
        res::PrimalDualSlack{UPLO, T, J},
        rhs::PrimalDualSlack{UPLO, T, J},
        pd1::PrimalDualSlack{UPLO, T, J},
        itr::PrimalDualSlack{UPLO, T, J},
        q::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        μ::T,
        scaling::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    if SCALE
        x = itr.primal
        p = itr.primal
        d = pd1.primal
    else
        x = q
        p = itr.slack
        d = pd1.slack
    end

    fill!(rhs, zero(T))

    copyto!(rhs.primal, d)
    copyto!(res.primal, d)

    hessian!(space, L, x, rhs.primal, scaling)
    thirdorder!(space, wrk.primal, L, x, res.primal, scaling)

    dot1 = symdot(p.X, wrk.primal.X)
    dot2 = symdot(d.X, rhs.primal.X)
    viol = abs(dot1 - dot2) / (sqrt(eps(T)) + abs(dot2))

    if viol < 1e-4
        axpy!(one(T), wrk.primal, rhs.primal)
    end

    lmul!(μ, rhs.primal)
    return
end

function centering_toa_rhs!(
        space::Workspace{T, J},
        wrk::PrimalDualSlack{UPLO, T, J},
        res::PrimalDualSlack{UPLO, T, J},
        rhs::PrimalDualSlack{UPLO, T, J},
        cd1::PrimalDualSlack{UPLO, T, J},
        itr::PrimalDualSlack{UPLO, T, J},
        q::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        μ::T,
        scaling::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    if SCALE
        x = itr.primal
        p = itr.primal
        d = cd1.primal
    else
        x = q
        p = itr.slack
        d = cd1.slack
    end

    fill!(rhs, zero(T))

    copyto!(rhs.primal, d)
    copyto!(res.primal, d)

    hessian!(space, L, x, rhs.primal, scaling)
    thirdorder!(space, wrk.primal, L, x, res.primal, scaling)

    dot1 = symdot(p.X, wrk.primal.X)
    dot2 = symdot(d.X, rhs.primal.X)
    viol = abs(dot1 - dot2) / (sqrt(eps(T)) + abs(dot2))

    if viol < 1e-4
        copyto!(rhs.primal, wrk.primal)
        lmul!(μ, rhs.primal)
        return true
    end

    return false
end

function linesearch_combined!(
        space::Workspace{T, J},
        wrk::PrimalDualSlack{UPLO, T, J},
        res::PrimalDualSlack{UPLO, T, J},
        itr::PrimalDualSlack{UPLO, T, J},
        pd1::PrimalDualSlack{UPLO, T, J},
        pd2::PrimalDualSlack{UPLO, T, J},
        cd1::PrimalDualSlack{UPLO, T, J},
        cd2::PrimalDualSlack{UPLO, T, J},
        r::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        prox_bound::T,
        scaling::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    n = ncl(L)
    copyto!(wrk, itr)

    if SCALE
        p = itr.primal
        q = itr.slack
    else
        p = itr.slack
        q = itr.primal
    end

    for α in ALPHA_SCHED
        β = one(T) - α

        copyto!(itr, wrk)

        axpy!(α,     pd1, itr)
        axpy!(β,     cd1, itr)
        axpy!(α * α, pd2, itr)
        axpy!(β * β, cd2, itr)

        ispositive(p.τ) || continue
        ispositive(q.τ) || continue

        μ = dot(p, q) / (n + 1)
        ispositive(μ) || continue

        ρ = (p.τ * q.τ) / μ
        ρ < 0.01 && continue
        ρ < one(T) - prox_bound && continue
        ρ > one(T) + prox_bound && continue

        σ = symdot(p.X, q.X) / (μ * n)
        σ < 0.01 && continue
        σ < one(T) - prox_bound / sqrt(n) && continue
        σ > one(T) + prox_bound / sqrt(n) && continue

        factorize!(space, L, q, flip(scaling)) || continue
        factorize!(space, L, p, scaling) || continue

        gradient!(space, r, L, p, scaling)
        prox = proximity!(space, res, itr, r, L, μ, scaling)

        if prox <= prox_bound
            return CONTINUE, α, prox
        end
    end

    copyto!(itr, wrk)
    return NUMERICAL_FAILURE, zero(T), zero(T)
end

function combined_phase!(
        space::Workspace{T, J},
        cache::KKT{T},
        itr::PrimalDualSlack{UPLO, T, J},
        pd1::PrimalDualSlack{UPLO, T, J},
        pd2::PrimalDualSlack{UPLO, T, J},
        cd1::PrimalDualSlack{UPLO, T, J},
        cd2::PrimalDualSlack{UPLO, T, J},
        rhs::PrimalDualSlack{UPLO, T, J},
        wrk::PrimalDualSlack{UPLO, T, J},
        res::PrimalDualSlack{UPLO, T, J},
        x::Primal{UPLO, T, J},
        q::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        problem::Problem{T, J},
        μ::T,
        prox_bound::T,
        min_res_norm::T,
        scaling::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    if SCALE
        z = itr.slack
    else
        z = itr.primal
    end

    copyto!(rhs.primal, z)
    lmul!(-one(T), rhs.primal)

    solve_kkt!(space, cache, pd1, rhs, x, L, problem, μ, scaling)
    refine_kkt!(space, cache, wrk, res, pd1, itr, rhs, q, L, problem, μ, min_res_norm, scaling)

    prediction_toa_rhs!(space, wrk, res, rhs, pd1, itr, q, L, μ, scaling)
    solve_kkt!(space, cache, pd2, rhs, x, L, problem, μ, scaling)
    refine_kkt!(space, cache, wrk, res, pd2, itr, rhs, q, L, problem, μ, min_res_norm, scaling)

    corrector_rhs!(rhs, itr, q, μ, scaling)
    solve_kkt!(space, cache, cd1, rhs, x, L, problem, μ, scaling)
    refine_kkt!(space, cache, wrk, res, cd1, itr, rhs, q, L, problem, μ, min_res_norm, scaling)

    flag = centering_toa_rhs!(space, wrk, res, rhs, cd1, itr, q, L, μ, scaling)

    if flag
        solve_kkt!(space, cache, cd2, rhs, x, L, problem, μ, scaling)
        refine_kkt!(space, cache, wrk, res, cd2, itr, rhs, q, L, problem, μ, min_res_norm, scaling)
    else
        fill!(cd2, zero(T))
    end

    status, α, prox = linesearch_combined!(
        space, wrk, res, itr, pd1, pd2, cd1, cd2,
        q, L, prox_bound, scaling)

    return status, α, prox
end

function check_termination(state::State{T}, settings::Settings{T}) where {T}
    τ = state.τ
    pres = state.pres / τ
    dres = state.dres / τ
    g = gap(state) / τ^2

    if dres <= settings.feas && pres <= settings.feas && (g <= settings.abs_opt || relgap(state) <= settings.rel_opt)
        return OPTIMAL
    elseif state.pinf <= settings.infeas && τ <= settings.tau_infeas
        return PRIMAL_INFEASIBLE
    elseif state.dinf <= settings.infeas && τ <= settings.tau_infeas
        return DUAL_INFEASIBLE
    elseif max(τ, state.κ) <= settings.illposed
        return ILL_POSED
    elseif state.nitr == settings.iter_limit
        return ITERATION_LIMIT
    end

    return CONTINUE
end

function check_termination_near!(state::State{T}, settings::Settings{T}) where {T}
    f = settings.near_factor
    τ = state.τ
    pres = state.pres / τ
    dres = state.dres / τ
    g = gap(state) / τ^2

    if state.status in (SLOW_PROGRESS, ITERATION_LIMIT, NUMERICAL_FAILURE)
        if dres <= f * settings.feas && pres <= f * settings.feas && (g <= f * settings.abs_opt || relgap(state) <= f * settings.rel_opt)
            state.status = NEAR_OPTIMAL
        elseif state.pinf <= f * settings.infeas && τ <= f * settings.tau_infeas
            state.status = NEAR_PRIMAL_INFEASIBLE
        elseif state.dinf <= f * settings.infeas && τ <= f * settings.tau_infeas
            state.status = NEAR_DUAL_INFEASIBLE
        elseif max(τ, state.κ) <= f * settings.illposed
            state.status = NEAR_ILL_POSED
        end
    end

    return state
end

function solve_loop!(
        state::State{T},
        prev::State{T},
        space::Workspace{T, J},
        cache::KKT{T},
        itr::PrimalDualSlack{UPLO, T, J},
        pd1::PrimalDualSlack{UPLO, T, J},
        pd2::PrimalDualSlack{UPLO, T, J},
        cd1::PrimalDualSlack{UPLO, T, J},
        cd2::PrimalDualSlack{UPLO, T, J},
        rhs::PrimalDualSlack{UPLO, T, J},
        wrk::PrimalDualSlack{UPLO, T, J},
        res::PrimalDualSlack{UPLO, T, J},
        q::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        problem::Problem{T, J},
        settings::Settings{T},
        scaling::Val{SCALE},
        resy0::T,
        resx0::T,
    ) where {UPLO, T, J, SCALE}

    update_state!(state, prev, itr, rhs, res, problem, settings, resy0, resx0)

    settings.verbose && print_loop(state)

    min_res_norm = T(1e-4) * max(state.pres, state.dres, abs(state.pobj - state.dobj + state.κ))

    if state.status == CONTINUE
        if SCALE
            p = itr.primal
            x = itr.primal
        else
            p = itr.slack
            x = q
        end

        flag = factorize!(space, L, p, scaling)

        if flag
            gradient!(space, q, L, p, scaling)
            build_kkt!(space, cache, x, res.primal, L, problem, state.μ, scaling)

            state.status, step, state.prox = combined_phase!(
                space, cache,
                itr, pd1, pd2, cd1, cd2, rhs, wrk, res,
                x, q, L,
                problem,
                state.μ, settings.prox_bound, min_res_norm, scaling)
        else
            state.status = NUMERICAL_FAILURE
        end
    end
end

# ===== Verbose printing =====

function show_banner(io::IO, indent::Int)
    pad = " "^indent
    w = 93
    println(io, pad, "-"^w)
    println(io, pad, center("Madeline.jl", w))
    println(io, pad, center("(c) Richard Samuelson", w))
    println(io, pad, center("University of Florida, 2025", w))
    println(io, pad, "-"^w)
    println(io)
    return
end

function center(s::String, w::Int)
    pad = w - length(s)
    left = pad ÷ 2
    right = pad - left
    return " "^left * s * " "^right
end

# ===== CommonSolve Interface =====

function CommonSolve.init(::Type{Solver}, problem::Problem)
    return Solver(problem)
end

function CommonSolve.solve(problem::Problem{T}; settings::Settings{T}=Settings{T}()) where {T}
    solver = Solver(problem)
    return solve!(solver; settings)
end

function CommonSolve.solve!(solver::Solver{UPLO, T, J}; settings::Settings{T}=Settings{T}()) where {UPLO, T, J}
    if settings.verbose
        show_banner(stdout, 0)
        println("problem:")
        show_problem(stdout, solver.problem, 2)
        println()
        println()
        println("settings:")
        show_settings(stdout, settings, 2)
        println()
        println()
        print_header()
    end

    while solver.curr.status == CONTINUE
        step!(solver; settings)
    end

    return stop!(solver; settings)
end

function CommonSolve.step!(solver::Solver{UPLO, T, J}; settings::Settings{T}=Settings{T}()) where {UPLO, T, J}
    if settings.scaling
        solve_loop!(
            solver.curr, solver.prev,
            solver.space, solver.cache,
            solver.itr, solver.pd1, solver.pd2, solver.cd1, solver.cd2,
            solver.rhs, solver.wrk, solver.res,
            solver.q, solver.L,
            solver.problem,
            settings, Val(true),
            solver.resy0, solver.resx0)
    else
        solve_loop!(
            solver.curr, solver.prev,
            solver.space, solver.cache,
            solver.itr, solver.pd1, solver.pd2, solver.cd1, solver.cd2,
            solver.rhs, solver.wrk, solver.res,
            solver.q, solver.L,
            solver.problem,
            settings, Val(false),
            solver.resy0, solver.resx0)
    end

    copyto!(solver.prev, solver.curr)
    solver.curr.nitr += 1

    return solver.curr.status
end

function stop!(solver::Solver{UPLO, T, J}; settings::Settings{T}=Settings{T}()) where {UPLO, T, J}
    check_termination_near!(solver.curr, settings)

    if settings.verbose
        print_terminated(solver.curr)
    end

    return Result(solver)
end
