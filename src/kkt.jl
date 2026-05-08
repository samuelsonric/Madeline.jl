# KKT: rank-2 form of the KKT system
#
# Keeps τ as a coordinate of K' throughout (no analytic elimination).
# The homogenization appears as a rank-2 skew lift on a symmetric base.
#
# Storage:
#   - chol: pivoted Cholesky factor of S̄ = A₂H₂₂⁻¹A₂* + h_τ²bb*
#   - Γ: half-solved cache vectors [u v] where u = ωL⁻¹b, v = L⁻¹(Aη)
#   - Σ: 2×2 capacitance matrix
#   - γ: 2-vector for Σ⁻¹ξ solution
#   - U, V, indices, idxmap: sparse constraint infrastructure
struct KKT{T, I}
    chol::Chol{T}
    Γ::FMatrix{T}                             # [u v] half-solved vectors (m × 2)
    Σ::FMatrix{T}                             # 2×2 capacitance
    γ::FVector{T}                             # Σ⁻¹ξ solution
    U::FMatrix{T}                             # workspace for sparse constraints (n × nrhs)
    V::FMatrix{T}                             # W^T W for sparse constraints (nrhs × nrhs)
    indices::FVector{I}                       # touched indices
    idxmap::FVector{I}                        # index map
end

function touched(A::SparseMatrixCSC{T, I}, k::I) where {T, I}
    n = convert(I, isqrt(size(A, 1)))
    m = convert(I, size(A, 2))

    indices = FVector{I}(undef, n)
    idxmap = FVector{I}(undef, n)

    fill!(indices, zero(I))
    fill!(idxmap, zero(I))

    @inbounds for c in k + one(I):m
        for p in nzrange(A, c)
            i, j = cart(n, rowvals(A)[p])
            idxmap[i] = one(I)
            idxmap[j] = one(I)
        end
    end

    nrhs = zero(I)

    @inbounds for i in oneto(n)
        if !iszero(idxmap[i])
            idxmap[i] = nrhs += one(I)
            indices[nrhs] = i
        end
    end

    return indices, idxmap, nrhs
end

function KKT{T}(problem::Problem{T, I}) where {T, I}
    A = problem.A
    S = problem.S
    k = problem.k
    n = size(S, 1)
    m = size(A, 2)
    indices, idxmap, nrhs = touched(A, k)
    chol = Chol{T}(m)
    Γ = FMatrix{T}(undef, m, 2)
    Σ = FMatrix{T}(undef, 2, 2)
    γ = FVector{T}(undef, 2)
    U = FMatrix{T}(undef, n, nrhs)
    V = FMatrix{T}(undef, nrhs, nrhs)
    return KKT{T, I}(chol, Γ, Σ, γ, U, V, indices, idxmap)
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

function build_schur_sparse_impl!(
        cache::KKT{T, I},
        problem::Problem{T, I},
        ω::T,
    ) where {T, I}
    H = cache.chol.L
    V = cache.V
    A = problem.A
    b = problem.b
    k = problem.k
    idxmap = cache.idxmap

    n = convert(I, isqrt(size(A, 1)))
    m = convert(I,       size(A, 2))

    @inbounds for cj in k + one(I):m
        for ci in cj:m
            H[ci, cj] = ω * b[ci] * b[cj]
        end
    end

    @inbounds for cj in k + one(I):m
        for pj in nzrange(A, cj)
            xj, yj = cart(n, rowvals(A)[pj])

            αj = nonzeros(A)[pj]

            if xj != yj
                αj *= two(T)
            end

            xv = idxmap[xj]
            yv = idxmap[yj]

            for ci in cj:m
                Δij = zero(T)

                for pi in nzrange(A, ci)
                    xi, yi = cart(n, rowvals(A)[pi])

                    αi = nonzeros(A)[pi]

                    xu = idxmap[xi]
                    yu = idxmap[yi]

                    Δij += αi * V[xu, xv] * V[yu, yv]

                    if xi != yi
                        Δij += αi * V[yu, xv] * V[xu, yv]
                    end
                end

                H[ci, cj] += Δij * αj
            end
        end
    end

    return H
end

function build_schur_sparse!(
        space::Workspace{T, J},
        cache::KKT{T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        problem::Problem{T, J},
        ω::T,
    ) where {UPLO, T, J}
    fill!(cache.U, zero(T))

    for k in axes(cache.U, 2)
        cache.U[cache.indices[k], k] = one(T)
    end

    div_impl!(cache.U, space.Mptr, space.Mval, space.Fval, L, Val(:N), Val(:N), Val(UPLO))
    syrk!(Val(:L), Val(:T), one(T), cache.U, zero(T), cache.V)
    symmtri!(cache.V, Val(:L))
    build_schur_sparse_impl!(cache, problem, ω)
    return
end

function build_schur!(
        space::Workspace{T, J},
        cache::KKT{T, J},
        x::Primal{UPLO, T, J},
        w::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        problem::Problem{T, J},
    ) where {UPLO, T, J}
    m = size(cache.chol.L, 1)

    for i in oneto(problem.k)
        copytopacked!(w, problem.A, problem.indices_primal, problem.b, i)
        hessian!(space, L, x, w, Val(false))

        for j in i:m
            cache.chol.L[j, i] = dotpacked(w, problem.A, problem.indices_primal, problem.b, j)
        end
    end

    if problem.k < m
        build_schur_sparse!(space, cache, L, problem, x.τ^2)
    end

    factorize!(cache.chol)
    return
end

# ===== KKT system =====

function build_kkt!(
        space::Workspace{T, J},
        cache::KKT{T, J},
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

    u = @view cache.Γ[:, 1]
    v = @view cache.Γ[:, 2]

    copyto!(u, problem.b)
    rmul!(u, w.τ)
    apply_constraint!(problem.A, problem.indices_primal, w.X, v, one(T), zero(T), Val(true))
    ldiv_fwd!(cache.chol, cache.Γ)

    syrk!(Val(:L), Val(:T), -one(T), cache.Γ, zero(T), cache.Σ)
    symmtri!(cache.Σ, Val(:L))

    cache.Σ[1, 1] += w.τ
    cache.Σ[2, 1] += σ
    cache.Σ[1, 2] -= σ
    cache.Σ[2, 2] += symdot(w.X, problem.C)

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
        cache::KKT{T, J},
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
    ldiv_fwd!(cache.chol, dir.dual)

    cache.γ[1] =  dir.primal.τ
    cache.γ[2] = -symdot(problem.C, dir.primal.X)

    gemv!(Val(:T), one(T), cache.Γ, dir.dual, one(T), cache.γ)
    solve2x2!(cache.Σ, cache.γ)
    gemv!(Val(:N), one(T), cache.Γ, cache.γ, one(T), dir.dual)

    copyto!(dir.slack, rhs.slack)
    ldiv_bwd!(cache.chol, dir.dual)
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
        cache::KKT{T, J},
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
