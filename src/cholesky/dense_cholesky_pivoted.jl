mutable struct DenseCholeskyPivoted{T} <: AbstractCholesky{T}
    const L::FMatrix{T}
    const perm::FVector{BlasInt}
    const work::FVector{T}
    const temp::FVector{T}
    rank::Int
    const del_static::T
    const tol_dynamic::T
    const del_dynamic::T
end

function DenseCholeskyPivoted{T}(m::Integer, del_static::T, tol_dynamic::T, del_dynamic::T) where {T}
    L = FMatrix{T}(undef, m, m)
    perm = FVector{BlasInt}(undef, m)
    work = FVector{T}(undef, 2m)
    temp = FVector{T}(undef, m)
    return DenseCholeskyPivoted(L, perm, work, temp, 0, del_static, tol_dynamic, del_dynamic)
end

function factorize!(chol::DenseCholeskyPivoted{T}) where {T}
    delta = chol.del_dynamic
    epsilon = chol.tol_dynamic

    if !ispositive(epsilon)
        _, chol.rank = pstrf!(Val(:L), chol.work, chol.L, chol.perm)
    else
        _, chol.rank = pstrf!(Val(:L), chol.work, chol.L, chol.perm, DynamicRegularization(; delta, epsilon))
    end

    return true
end

function setzero!(chol::DenseCholeskyPivoted{T}) where {T}
    fill!(chol.L, zero(T))

    @inbounds for i in diagind(chol.L)
        chol.L[i] = chol.del_static
    end

    return chol
end

function addclique!(chol::DenseCholeskyPivoted{T}, W::AbstractMatrix{T}, clique::AbstractVector{I}) where {T, I}
    @inbounds for jloc in eachindex(clique)
        cj = clique[jloc]

        for iloc in jloc:length(clique)
            ci = clique[iloc]
            chol.L[ci, cj] += W[iloc, jloc]
        end
    end

    return chol
end

function ldiv_fwd!(chol::DenseCholeskyPivoted{T}, b::AbstractVector{T}) where {T}
    @inbounds for i in 1:chol.rank
        chol.temp[i] = b[chol.perm[i]]
    end

    @inbounds for i in 1:chol.rank
        b[i] = chol.temp[i]
    end

    @inbounds for i in chol.rank + 1:length(b)
        b[i] = zero(T)
    end

    if ispositive(chol.rank)
        Lr = view(chol.L, 1:chol.rank, 1:chol.rank)
        br = view(b,      1:chol.rank)
        trsv!(Val(:L), Val(:N), Val(:N), Lr, br)
    end

    return b
end

function ldiv_fwd!(chol::DenseCholeskyPivoted{T}, B::AbstractMatrix{T}) where {T}
    @inbounds for j in axes(B, 2)
        for i in 1:chol.rank
            chol.temp[i] = B[chol.perm[i], j]
        end

        for i in 1:chol.rank
            B[i, j] = chol.temp[i]
        end

        for i in chol.rank + 1:size(B, 1)
            B[i, j] = zero(T)
        end
    end

    if ispositive(chol.rank)
        Lr = view(chol.L, 1:chol.rank, 1:chol.rank)
        Br = view(B,      1:chol.rank, :)
        trsm!(Val(:L), Val(:L), Val(:N), Val(:N), one(T), Lr, Br)
    end

    return B
end

function ldiv_bwd!(chol::DenseCholeskyPivoted, b::AbstractVector)
    if ispositive(chol.rank)
        Lr = view(chol.L, 1:chol.rank, 1:chol.rank)
        br = view(b,      1:chol.rank)
        trsv!(Val(:L), Val(:T), Val(:N), Lr, br)
    end

    @inbounds for i in eachindex(b)
        chol.temp[chol.perm[i]] = b[i]
    end

    copyto!(b, chol.temp)
    return b
end
