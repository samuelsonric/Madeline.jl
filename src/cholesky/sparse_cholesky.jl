struct SparseCholesky{T, I} <: AbstractCholesky{T}
    F::FChordalCholesky{:L, T, I}
    W::FMatrix{T}         # workspace for permuted W matrix
    prm::FVector{I}       # workspace for sorting permutation
    ivp::FVector{I}       # workspace for inverse permutation
    Mptr::FVector{I}      # workspace for ldiv
    Mval::FVector{T}      # workspace for ldiv
    Fval::FVector{T}      # workspace for ldiv
    temp::FVector{T}      # workspace for permuted rhs
end

function SparseCholesky{T, I}(pattern::SparseMatrixCSC, k::Integer) where {T, I}
    F = FChordalCholesky{:L, T, I}(pattern)
    n = size(F, 1)

    W = FMatrix{T}(undef, k, k)
    prm = FVector{I}(undef, k)
    ivp = FVector{I}(undef, k)
    Mptr = FVector{I}(undef, F.L.S.nMptr)
    Mval = FVector{T}(undef, max(F.L.S.nMval, F.L.S.nNval * 2))
    Fval = FVector{T}(undef, max(F.L.S.nFval * F.L.S.nFval, F.L.S.nFval * 2))
    temp = FVector{T}(undef, n)

    return SparseCholesky(F, W, prm, ivp, Mptr, Mval, Fval, temp)
end

function setzero!(chol::SparseCholesky{T}) where {T}
    fill!(chol.F, zero(T))
    return chol
end

function factorize!(chol::SparseCholesky{T}) where {T}
    info = chol_impl!(chol.Mptr, chol.Mval, chol.Fval, chol.F.L)
    return iszero(info)
end

function addclique!(chol::SparseCholesky{T, I}, A::AbstractMatrix{T}, clique::AbstractVector{I}) where {T, I}
    F = chol.F
    n = length(clique)

    prm = view(chol.prm, oneto(n))
    ivp = view(chol.ivp, oneto(n))
    W = view(chol.W, oneto(n), oneto(n))

    @inbounds for i in eachindex(clique)
        ivp[i] = F.invp[clique[i]]
    end

    sortperm!(prm, ivp)

    @inbounds for i in oneto(n)
        ivp[prm[i]] = i
    end

    sympermute!(W, A, ivp, 'L', 'L')

    @inbounds for i in oneto(n)
        prm[ivp[i]] = F.invp[clique[i]]
    end

    add_clique_impl!(F.L, W, prm)

    return chol
end

function add_clique_impl!(L::FChordalTriangular{:N, :L, T, I}, W::AbstractMatrix{T}, ind::AbstractVector{I}) where {T, I}
    n = convert(I, length(ind))

    f = L.S.idx[ind[1]]

    D, res = diagblock(L, f)
    B, sep = offdblock(L, f)

    rlo = first(res)
    rhi = last(res)

    if !isempty(sep)
        slo = first(sep)
        shi = last(sep)
    end

    for k in oneto(n)
        v = ind[k]

        if rhi < v
            f = L.S.idx[v]

            D, res = diagblock(L, f)
            B, sep = offdblock(L, f)

            rlo = first(res)
            rhi = last(res)

            if !isempty(sep)
                slo = first(sep)
                shi = last(sep)
            end
        end

        vloc = v - rlo + one(I)
        sloc = one(I)

        for j in k:n
            w = ind[j]

            if w <= rhi
                wloc = w - rlo + one(I)
                D[wloc, vloc] += W[j, k]
            elseif !isempty(sep) && w >= slo && w <= shi
                while sep[sloc] < w
                    sloc += one(I)
                end

                B[sloc, vloc] += W[j, k]
            else
                break
            end
        end
    end

    return
end

function ldiv_fwd!(chol::SparseCholesky{T}, b::AbstractVector{T}) where {T}
    F = chol.F
    n = length(b)

    @inbounds for i in 1:n
        chol.temp[i] = b[F.perm[i]]
    end

    @inbounds for i in 1:n
        b[i] = chol.temp[i]
    end

    div_impl!(b, chol.Mptr, chol.Mval, chol.Fval, F.L, Val(:N), Val(:N), F.L.uplo)

    return b
end

function ldiv_fwd!(chol::SparseCholesky{T}, B::AbstractMatrix{T}) where {T}
    F = chol.F

    @inbounds for j in axes(B, 2)
        for i in axes(B, 1)
            chol.temp[i] = B[F.perm[i], j]
        end

        for i in axes(B, 1)
            B[i, j] = chol.temp[i]
        end
    end

    div_impl!(B, chol.Mptr, chol.Mval, chol.Fval, F.L, Val(:N), Val(:N), F.L.uplo)

    return B
end

function ldiv_bwd!(chol::SparseCholesky{T}, b::AbstractVector{T}) where {T}
    F = chol.F

    div_impl!(b, chol.Mptr, chol.Mval, chol.Fval, F.L, Val(:C), Val(:N), F.L.uplo)

    @inbounds for i in eachindex(b)
        chol.temp[F.perm[i]] = b[i]
    end

    copyto!(b, chol.temp)

    return b
end
