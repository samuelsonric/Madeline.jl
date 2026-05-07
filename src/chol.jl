mutable struct Chol{T}
    const L::FMatrix{T}
    const perm::FVector{BlasInt}
    const work::FVector{T}
    const temp::FVector{T}
    rank::Int
end

function Chol{T}(m::Integer) where {T}
    L = FMatrix{T}(undef, m, m)
    perm = FVector{BlasInt}(undef, m)
    work = FVector{T}(undef, 2m)
    temp = FVector{T}(undef, m)
    return Chol(L, perm, work, temp, 0)
end

function factorize!(chol::Chol{T}) where {T}
    _, chol.rank = pstrf!(Val(:L), chol.work, chol.L, chol.perm, -one(T))
    return chol
end

function ldiv_fwd!(chol::Chol, b::AbstractVector)
    @inbounds for i in eachindex(b)
        chol.temp[i] = b[chol.perm[i]]
    end

    copyto!(b, chol.temp)

    @inbounds for i in chol.rank+1:length(b)
        b[i] = zero(eltype(b))
    end

    if ispositive(chol.rank)
        Lr = view(chol.L, 1:chol.rank, 1:chol.rank)
        br = view(b,      1:chol.rank)
        trsv!(Val(:L), Val(:N), Val(:N), Lr, br)
    end

    return b
end

function ldiv_bwd!(chol::Chol, b::AbstractVector)
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
