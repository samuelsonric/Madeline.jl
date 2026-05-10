struct DenseCholesky{T} <: AbstractCholesky{T}
    L::FMatrix{T}
    temp::FVector{T}
end

function DenseCholesky{T}(m::Integer) where {T}
    L = FMatrix{T}(undef, m, m)
    temp = FVector{T}(undef, m)
    return DenseCholesky(L, temp)
end

function factorize!(chol::DenseCholesky{T}) where {T}
    potrf!(Val(:L), chol.L)
    return chol
end

@propagate_inbounds function setfactorindex!(chol::DenseCholesky{T}, v::T, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(chol.L, i, j)
    @inbounds chol.L[i, j] = v
    return chol
end

@propagate_inbounds function addfactorindex!(chol::DenseCholesky{T}, v::T, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(chol.L, i, j)
    @inbounds chol.L[i, j] += v
    return chol
end

function setfactorzero!(chol::DenseCholesky{T}) where {T}
    fill!(chol.L, zero(T))
    return chol
end

function ldiv_fwd!(chol::DenseCholesky{T}, b::AbstractVector{T}) where {T}
    trsv!(Val(:L), Val(:N), Val(:N), chol.L, b)
    return b
end

function ldiv_fwd!(chol::DenseCholesky{T}, B::AbstractMatrix{T}) where {T}
    trsm!(Val(:L), Val(:L), Val(:N), Val(:N), one(T), chol.L, B)
    return B
end

function ldiv_bwd!(chol::DenseCholesky{T}, b::AbstractVector{T}) where {T}
    trsv!(Val(:L), Val(:T), Val(:N), chol.L, b)
    return b
end
