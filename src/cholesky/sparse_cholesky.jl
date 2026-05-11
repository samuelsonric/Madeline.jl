struct SparseCholesky{T, I} <: AbstractCholesky{T}
    F::FChordalCholesky{:L, T, I}
    prm::FVector{I}       # workspace for sorting permutation
    ivp::FVector{I}       # workspace for inverse permutation
    W::FMatrix{T}         # workspace for permuted W matrix
end

function setzero!(chol::SparseCholesky{T}) where {T}
    fill!(chol.F, zero(T))
    return chol
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
    S = L.S
    n = length(ind)

    lo = ind[1]
    hi = ind[n]

    flo = S.idx[lo]
    fhi = S.idx[hi]

    f = flo
    k = one(I)

    while f <= fhi
        D, res = diagblock(L, f)
        B, sep = offdblock(L, f)

        rlo = first(res)
        rhi = last(res)

        if !isempty(sep)
            slo = first(sep)
            shi = last(sep)
        end

        while k <= n && ind[k] <= rhi
            v = ind[k]

            if v >= rlo
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
                    end
                end
            end

            k += one(I)
        end

        f = S.pnt[f]
    end

    return L
end
