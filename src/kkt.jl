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
