struct SparseCholeskyPivoted{T, I} <: AbstractCholesky{T}
    F::FChordalCholesky{:L, T, I}
    W::FMatrix{T}         # workspace for permuted W matrix
    prm::FVector{I}       # workspace for sorting permutation
    ivp::FVector{I}       # workspace for inverse permutation
    Mptr::FVector{I}      # workspace for ldiv
    Mval::FVector{T}      # workspace for ldiv
    Fval::FVector{T}      # workspace for ldiv
    temp::FVector{T}      # workspace for permuted rhs
    piv::FVector{BlasInt} # workspace for pivots
    mval::FVector{I}      # workspace for pivoting
    fval::FVector{I}      # workspace for pivoting
    del_static::T
    tol_dynamic::T
    del_dynamic::T
end

function SparseCholeskyPivoted{T, I}(pattern::SparseMatrixCSC, k::Integer, del_static::T, tol_dynamic::T, del_dynamic::T) where {T, I}
    F = FChordalCholesky{:L, T, I}(pattern)
    n = size(F, 1)
    L = F.L

    W = FMatrix{T}(undef, k, k)
    prm = FVector{I}(undef, k)
    ivp = FVector{I}(undef, k)
    Mptr = FVector{I}(undef, L.S.nMptr)
    Mval = FVector{T}(undef, max(L.S.nMval, L.S.nNval * 2))
    Fval = FVector{T}(undef, max(L.S.nFval * L.S.nFval, L.S.nFval * 2))
    temp = FVector{T}(undef, n)
    piv = FVector{BlasInt}(undef, L.S.nFval)
    mval = FVector{I}(undef, L.S.nNval)
    fval = FVector{I}(undef, L.S.nFval)

    return SparseCholeskyPivoted(F, W, prm, ivp, Mptr, Mval, Fval, temp, piv, mval, fval, del_static, tol_dynamic, del_dynamic)
end

function setzero!(chol::SparseCholeskyPivoted{T}) where {T}
    axpby!(chol.del_static, I, zero(T), chol.F.L)
    return chol
end

function factorize!(chol::SparseCholeskyPivoted{T}) where {T}
    F = chol.F
    delta = chol.del_dynamic
    epsilon = chol.tol_dynamic

    if !ispositive(epsilon)
        chol_piv_impl!(chol.Mptr, chol.Mval, chol.Fval, chol.piv, chol.mval, chol.fval, F.L, F.perm, F.invp)
    else
        chol_piv_impl!(chol.Mptr, chol.Mval, chol.Fval, chol.piv, chol.mval, chol.fval, F.L, F.perm, F.invp, DynamicRegularization(; delta, epsilon))
    end

    return true
end

function addclique!(chol::SparseCholeskyPivoted{T, I}, A::AbstractMatrix{T}, clique::AbstractVector{I}) where {T, I}
    n = length(clique)
    prm = view(chol.prm, oneto(n))
    ivp = view(chol.ivp, oneto(n))
    W = view(chol.W, oneto(n), oneto(n))
    add_clique_impl!(chol.F, A, clique, W, prm, ivp)
    return chol
end

function ldiv_fwd!(chol::SparseCholeskyPivoted{T}, b::AbstractVector{T}) where {T}
    F = chol.F
    n = length(b)

    @inbounds for i in 1:n
        chol.temp[i] = b[F.perm[i]]
    end

    @inbounds for i in 1:n
        b[i] = chol.temp[i]
    end

    if !ispositive(chol.tol_dynamic)
        div_impl!(b, chol.Mptr, chol.Mval, chol.Fval, F.L, Val(:N), Val(:N), F.L.uplo)
    else
        div_piv_impl!(b, chol.Mptr, chol.Mval, chol.Fval, F.L, Val(:N), Val(:N), F.L.uplo)
    end

    return b
end

function ldiv_fwd!(chol::SparseCholeskyPivoted{T}, B::AbstractMatrix{T}) where {T}
    F = chol.F

    @inbounds for j in axes(B, 2)
        for i in axes(B, 1)
            chol.temp[i] = B[F.perm[i], j]
        end

        for i in axes(B, 1)
            B[i, j] = chol.temp[i]
        end
    end

    if !ispositive(chol.tol_dynamic)
        div_impl!(B, chol.Mptr, chol.Mval, chol.Fval, F.L, Val(:N), Val(:N), F.L.uplo)
    else
        div_piv_impl!(B, chol.Mptr, chol.Mval, chol.Fval, F.L, Val(:N), Val(:N), F.L.uplo)
    end

    return B
end

function ldiv_bwd!(chol::SparseCholeskyPivoted{T}, b::AbstractVector{T}) where {T}
    F = chol.F

    if !ispositive(chol.tol_dynamic)
        div_impl!(b, chol.Mptr, chol.Mval, chol.Fval, F.L, Val(:C), Val(:N), F.L.uplo)
    else
        div_piv_impl!(b, chol.Mptr, chol.Mval, chol.Fval, F.L, Val(:C), Val(:N), F.L.uplo)
    end

    @inbounds for i in eachindex(b)
        chol.temp[F.perm[i]] = b[i]
    end

    copyto!(b, chol.temp)

    return b
end
