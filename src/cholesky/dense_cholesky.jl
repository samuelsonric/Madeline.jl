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
    info = potrf!(Val(:L), chol.L)
    return iszero(info)
end

function setzero!(chol::DenseCholesky{T}, shift::T) where {T}
    fill!(chol.L, zero(T))

    @inbounds for i in diagind(chol.L)
        chol.L[i] = shift
    end

    return chol
end

function addclique!(chol::DenseCholesky{T}, W::AbstractMatrix{T}, clique::AbstractVector{I}) where {T, I}
    @inbounds for jloc in eachindex(clique)
        cj = clique[jloc]

        for iloc in jloc:length(clique)
            ci = clique[iloc]
            chol.L[ci, cj] += W[iloc, jloc]
        end
    end

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
