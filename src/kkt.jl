# KKT: rank-2 form of the KKT system
#
# Keeps τ as a coordinate of K' throughout (no analytic elimination).
# The homogenization appears as a rank-2 skew lift on a symmetric base.
#
# Storage:
#   - chol: pivoted Cholesky factor of S₀ = A₂H₂₂⁻¹A₂*
#   - Γ: half-solved cache vectors [u₀ c₀] where u₀ = L⁻¹b, c₀ = L⁻¹(Aη)
#   - U, V: small workspaces for sparse constraint Schur complement
#                 (sized for largest connected component, reused via OffsetArrays)
#   - Σ: 2×2 capacitance matrix
#   - γ: 2-vector for Σ⁻¹ξ solution
#   - ρ: ⟨u₀, c₀⟩
mutable struct KKT{T, Chol <: AbstractCholesky{T}}
    const chol::Chol
    const U::FMatrix{T}              # workspace for sparse constraints (max_cc_rows × max_rhs_per_cc)
    const V::FMatrix{T}              # W^T W workspace (max_rhs_per_cc × max_rhs_per_cc)
    const Γ::FMatrix{T}                 # [u₀ c₀] after build_kkt!
    const Σ::FMatrix{T}                 # 2×2 capacitance
    const γ::FVector{T}                 # Σ⁻¹ξ solution
    ρ::T                                # ⟨u₀, c₀⟩
end

function KKT{T}(problem::Problem{T}) where {T}
    m = size(problem.A, 2)

    chol = DenseCholeskyPivoted{T}(m)
    U = FMatrix{T}(undef, problem.max_cc_rows, problem.max_rhs_per_cc)
    V = FMatrix{T}(undef, problem.max_rhs_per_cc, problem.max_rhs_per_cc)
    Γ = FMatrix{T}(undef, m, 2)
    Σ = FMatrix{T}(undef, 2, 2)
    γ = FVector{T}(undef, 2)
    return KKT(chol, U, V, Γ, Σ, γ, zero(T))
end

# ===== Schur complement =====

function build_gram!(cache::KKT, A::SparseMatrixCSC{T, I}) where {T, I}
    H = cache.chol.L
    n = isqrt(size(A, 1))
    m = size(A, 2)

    @inbounds for j in oneto(m)
        pjlo = A.colptr[j]
        pjhi = A.colptr[j + 1] - one(I)

        for i in j:m
            pi   = A.colptr[i]
            pihi = A.colptr[i + 1] - one(I)

            pj = pjlo

            Hij = zero(T)

            while pi <= pihi && pj <= pjhi
                ri = rowvals(A)[pi]
                rj = rowvals(A)[pj]

                if ri == rj
                    xi, yi = cart(n, ri)
                    Hij += Δ = nonzeros(A)[pi] * nonzeros(A)[pj]

                    if xi != yi
                        Hij += conj(Δ)
                    end

                    pi += 1
                    pj += 1
                elseif ri < rj
                    pi += 1
                else
                    pj += 1
                end
            end

            H[i, j] = Hij
        end
    end

    factorize!(cache.chol)
    return
end

function schur_entry_sparse(
        V::AbstractMatrix{T},
        A::SparseMatrixCSC{T, I},
        idxbwd::FVector{I},
        irange::AbstractRange{I},
        jrange::AbstractRange{I},
    ) where {T, I}
    n = convert(I, isqrt(size(A, 1)))
    rowval = rowvals(A)
    nzval = nonzeros(A)

    Hij = zero(T)

    @inbounds for pj in jrange
        xj, yj = cart(n, rowval[pj])

        αj = nzval[pj]

        if xj != yj
            αj *= two(T)
        end

        xv = idxbwd[xj]
        yv = idxbwd[yj]

        Δij = zero(T)

        for pi in irange
            xi, yi = cart(n, rowval[pi])

            αi = nzval[pi]

            xu = idxbwd[xi]
            yu = idxbwd[yi]

            Δij += αi * V[xu, xv] * V[yu, yv]

            if xi != yi
                Δij += αi * V[yu, xv] * V[xu, yv]
            end
        end

        Hij += Δij * αj
    end

    return Hij
end

function schur_column_dense!(
        space::Workspace{T, J},
        cache::KKT{T},
        X::ChordalTriangular{:N, UPLO, T, J},
        W::ChordalTriangular{:N, UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        problem::Problem{T, J},
        cj::J,
        kj::J,
        khi::J,
        frange::AbstractRange{J},
    ) where {UPLO, T, J}
    A = problem.A
    indices = problem.indices_primal
    cc_to_cons = problem.cc_to_cons
    cc_to_strt = problem.cc_to_strt
    cc_to_stop = problem.cc_to_stop
    chol = cache.chol

    pjstrt = targets(cc_to_strt)[kj]
    pjstop = targets(cc_to_stop)[kj]

    copytopacked!(W, A, indices, pjstrt:pjstop)
    hessian!(space, W, L, X, Val(false), frange)
    addfactorindex!(chol, dotpacked(W, A, indices, pjstrt:pjstop), cj, cj)

    for ki in kj + one(J):khi
        ci = targets(cc_to_cons)[ki]
        pistrt = targets(cc_to_strt)[ki]
        pistop = targets(cc_to_stop)[ki]
        addfactorindex!(chol, dotpacked(W, A, indices, pistrt:pistop), ci, cj)
    end

    return
end

function schur_column_sparse!(
        V::AbstractMatrix{T},
        chol::AbstractCholesky{T},
        problem::Problem{T, J},
        cj::J,
        kj::J,
        khi::J,
        srange::AbstractRange{J},
    ) where {T, J}
    # Wrap V with offset indices for idxbwd lookups
    Voff = OffsetArray(V, srange, srange)

    A = problem.A
    idxbwd = problem.idxbwd
    cc_to_cons = problem.cc_to_cons
    cc_to_strt = problem.cc_to_strt
    cc_to_stop = problem.cc_to_stop

    pjstrt = targets(cc_to_strt)[kj]
    pjstop = targets(cc_to_stop)[kj]

    addfactorindex!(chol, schur_entry_sparse(Voff, A, idxbwd, pjstrt:pjstop, pjstrt:pjstop), cj, cj)

    for ki in kj + one(J):khi
        ci = targets(cc_to_cons)[ki]
        pistrt = targets(cc_to_strt)[ki]
        pistop = targets(cc_to_stop)[ki]
        addfactorindex!(chol, schur_entry_sparse(Voff, A, idxbwd, pistrt:pistop, pjstrt:pjstop), ci, cj)
    end

    return
end

function process_cc!(
        space::Workspace{T, J},
        U::AbstractMatrix{T},
        V::AbstractMatrix{T},
        L::ChordalTriangular{:N, UPLO, T, J},
        frange::AbstractRange{J},
        rrange::AbstractRange{J},
    ) where {UPLO, T, J}
    # Wrap U with offset rows for div_impl!
    Uoff = OffsetArray(U, rrange, axes(U, 2))
    div_impl!(Uoff, space.Mptr, space.Mval, space.Fval, L, Val(:N), Val(:N), Val(UPLO), frange)
    # BLAS uses raw 1-based arrays
    syrk!(Val(:L), Val(:T), one(T), U, zero(T), V)
    symmtri!(V, Val(:L))
    return
end

function build_schur!(
        space::Workspace{T, J},
        cache::KKT{T},
        x::Primal{UPLO, T, J},
        w::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        problem::Problem{T, J},
    ) where {UPLO, T, J}
    k = problem.k
    chol = cache.chol
    cc_to_cons = problem.cc_to_cons

    setfactorzero!(chol)

    @timeit TIMER "schur" for cc in oneto(problem.ncc)
        fdsc = problem.frtptr[cc]
        root = problem.frtptr[cc + one(J)] - one(J)
        frange = fdsc:root

        klo = pointers(cc_to_cons)[cc]
        khi = pointers(cc_to_cons)[cc + one(J)] - one(J)

        slo = problem.idxptr[cc]
        shi = problem.idxptr[cc + one(J)] - one(J)
        ncols = shi - slo + one(J)

        if ispositive(ncols)
            rlo = L.S.res.ptr[fdsc]
            rhi = L.S.res.ptr[root + one(J)] - one(J)
            nrows = rhi - rlo + one(J)

            U = view(cache.U, oneto(nrows), oneto(ncols))
            V = view(cache.V, oneto(ncols), oneto(ncols))

            fill!(U, zero(T))

            for i in slo:shi
                U[problem.idxfwd[i] - rlo + one(J), i - slo + one(J)] = one(T)
            end

            process_cc!(space, U, V, L, frange, rlo:rhi)
        end

        for kj in klo:khi
            cj = targets(cc_to_cons)[kj]

            if cj <= k
                schur_column_dense!(space, cache, x.X, w.X, L, problem, cj, kj, khi, frange)
            else
                schur_column_sparse!(V, cache.chol, problem, cj, kj, khi, slo:shi)
            end
        end
    end

    @timeit TIMER "chol_factor" factorize!(cache.chol)
    return
end

# ===== KKT system =====

function build_kkt!(
        space::Workspace{T, J},
        cache::KKT{T},
        x::Primal{UPLO, T, J},
        w::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        problem::Problem{T, J},
        μ::T,
        ::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    if SCALE
        σ = μ
    else
        σ = inv(μ)
    end

    build_schur!(space, cache, x, w, L, problem)

    w.τ = one(T)
    copyto!(w.X, problem.C)
    hessian!(space, L, x, w, Val(false))

    Γ₁ = view(cache.Γ, :, 1)
    Γ₂ = view(cache.Γ, :, 2)

    copyto!(Γ₁, problem.b)
    apply_constraint!(problem.A, problem.indices_primal, w.X, Γ₂, one(T), zero(T), Val(true))
    ldiv_fwd!(cache.chol, cache.Γ)

    syrk!(Val(:L), Val(:T), one(T), cache.Γ, zero(T), cache.Σ)
    symmtri!(cache.Σ, Val(:L))

    Σ₁₁ = cache.Σ[1, 1]
    Σ₂₁ = cache.Σ[2, 1]
    Σ₂₂ = cache.Σ[2, 2]

    ω = w.τ
    α = ω / (one(T) + ω * Σ₁₁)

    cache.Σ[1, 1] =                          α
    cache.Σ[2, 1] =  σ                     - α * Σ₂₁
    cache.Σ[1, 2] = -σ                     - α * Σ₂₁
    cache.Σ[2, 2] = symdot(w.X, problem.C) + α * Σ₂₁ * Σ₂₁ - Σ₂₂

    cache.ρ = Σ₂₁
    return
end

# solve the KKT system
#
#    [  0   -b     Aᵀ ] [ dy ]   [ y' ]   [    ]
#    [ -bᵀ -μτ⁻²  -cᵀ ] [ dτ ] = [ κ' ] - [ τ' ]
#    [  A    c    -μH ] [ dx ]   [ z' ]   [ x' ]
#
function solve_kkt!(
        space::Workspace{T, J},
        cache::KKT{T},
        dir::PrimalDualSlack{UPLO, T, J},
        rhs::PrimalDualSlack{UPLO, T, J},
        x::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        problem::Problem{T, J},
        μ::T,
        ::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    if SCALE
        σ = μ
    else
        σ = inv(μ)
    end

    copyto!(dir.primal, rhs.slack)

    if SCALE
        axpy!(-one(T), rhs.primal, dir.primal)
    end

    hessian!(space, L, x, dir.primal, Val(false))

    if !SCALE
        axpy!(-σ, rhs.primal, dir.primal)
    end

    copyto!(dir.dual, rhs.dual)
    apply_constraint!(problem.A, problem.indices_primal, problem.b, dir.primal, dir.dual, one(T), σ, Val(true))

    Γ₁ = view(cache.Γ, :, 1)
    Γ₂ = view(cache.Γ, :, 2)

    cache.γ[1] =  dir.primal.τ
    cache.γ[2] = -symdot(problem.C, dir.primal.X)

    ldiv_fwd!(cache.chol, dir.dual)

    Δ = cache.Σ[1, 1] * dot(Γ₁, dir.dual)

    cache.γ[1] +=                     Δ
    cache.γ[2] += dot(Γ₂, dir.dual) - Δ * cache.ρ

    solve2x2!(cache.Σ, cache.γ)

    axpy!(cache.γ[2] * σ + dir.primal.τ, Γ₁, dir.dual)
    axpy!(cache.γ[2],                    Γ₂, dir.dual)

    ldiv_bwd!(cache.chol, dir.dual)

    copyto!(dir.slack, rhs.slack)
    apply_constraint!(problem.A, problem.indices_slack, problem.b, dir.slack, dir.dual, -one(T), one(T), Val(false))

    dir.slack.τ -= cache.γ[1]
    axpy_subset!(cache.γ[2], problem.C, dir.slack.X)

    copyto!(dir.primal, dir.slack)

    if SCALE
        axpy!(-one(T), rhs.primal, dir.primal)
    end

    hessian!(space, L, x, dir.primal, Val(false))

    if !SCALE
        axpy!(-σ, rhs.primal, dir.primal)
    end

    ldiv!(-σ, dir.primal)

    dir.primal.τ = cache.γ[2] # more precise
    return
end

function residual_kkt!(
        space::Workspace{T, J},
        res::PrimalDualSlack{UPLO, T, J},
        dir::PrimalDualSlack{UPLO, T, J},
        rhs::PrimalDualSlack{UPLO, T, J},
        problem::Problem{T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        x::Primal{UPLO, T, J},
        μ::T,
        scaling::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    residual!(res, dir, problem)
    axpy!(one(T), rhs.dual, res.dual)
    axpy!(one(T), rhs.slack, res.slack)

    if SCALE
        p = dir.primal
        q = dir.slack
    else
        p = dir.slack
        q = dir.primal
    end

    copyto!(res.primal, p)
    hessian!(space, L, x, res.primal, scaling)
    axpby!(one(T), rhs.primal, -μ, res.primal)
    axpy!(-one(T), q, res.primal)

    return norm(res)
end

function refine_kkt!(
        space::Workspace{T, J},
        cache::KKT{T},
        wrk::PrimalDualSlack{UPLO, T, J},
        res::PrimalDualSlack{UPLO, T, J},
        dir::PrimalDualSlack{UPLO, T, J},
        itr::PrimalDualSlack{UPLO, T, J},
        rhs::PrimalDualSlack{UPLO, T, J},
        q::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        problem::Problem{T, J},
        μ::T,
        min_res_norm::T,
        scaling::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    if SCALE
        x = itr.primal
    else
        x = q
    end

    ρ = residual_kkt!(space, res, dir, rhs, problem, L, x, μ, scaling)
    count = 0

    for _ in 1:MAX_REF_STEPS
        ρ ≤ min_res_norm && break
        2 ≤ count && break

        solve_kkt!(space, cache, wrk, res, x, L, problem, μ, scaling)
        axpy!(one(T), dir, wrk)

        ε = residual_kkt!(space, res, wrk, rhs, problem, L, x, μ, scaling)

        ε ≥ ρ && break

        copyto!(dir, wrk)

        if ε > REF_MIN_IMPR * ρ
            count += 1
        end

        ρ = ε
    end

    return
end
