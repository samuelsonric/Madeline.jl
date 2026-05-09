# ===== Problem data type =====

function trilinegraph(ve::FBipartiteGraph{V, E}, ev::FBipartiteGraph{V, E}) where {V, E}
    n = nv(ve)
    m = zero(E)
    marker = FVector{V}(undef, n)

    @inbounds for v in vertices(ve)
        marker[v] = zero(V)
    end

    @inbounds for v in vertices(ve)
        tag = v

        for w in neighbors(ve, v), x in neighbors(ev, w)
            if x > v && marker[x] < tag
                marker[x] = tag
                m += one(E)
            end
        end
    end

    target = FVector{E}(undef, m)
    pointer = FVector{V}(undef, n + one(V))
    @inbounds pointer[one(V)] = p = one(E)

    @inbounds for v in vertices(ve)
        tag = n + v

        for w in neighbors(ve, v), x in neighbors(ev, w)
            if x > v && marker[x] < tag
                marker[x] = tag
                target[p] = x
                p += one(E)
            end
        end

        pointer[v + one(V)] = p
    end

    return FBipartiteGraph{V, E}(n, n, m, pointer, target)
end

function constraint_graph(S::ChordalSymbolic{I}, A::SparseMatrixCSC{T, I}) where {T, I}
    m = convert(I, size(A, 2))
    n = convert(I, ncl(S))
    k = zero(I)

    cmp = FVector{I}(undef, nfr(S))

    for i in reverse(fronts(S))
        j = S.pnt[i]

        if iszero(j)
            cmp[i] = k += one(I)
        else
            cmp[i] = cmp[j]
        end
    end

    mark = FVector{I}(undef, k)
    fill!(mark, zero(I))

    p = zero(I)

    for c in oneto(m)
        jprv = zero(I)

        for q in nzrange(A, c)
            i, j = cart(n, rowvals(A)[q])

            r = cmp[S.idx[i]]

            if mark[r] != -c
                mark[r] = -c
                p += one(I)
            end

            if j != jprv
                r = cmp[S.idx[j]]

                if mark[r] != -c
                    mark[r] = -c
                    p += one(I)
                end
            end

            jprv = j
        end
    end

    graph = FBipartiteGraph{I, I}(k, m, p)

    p = zero(I)

    for c in oneto(m)
        pointers(graph)[c] = p + one(I)

        jprv = zero(I)

        for q in nzrange(A, c)
            i, j = cart(n, rowvals(A)[q])

            r = cmp[S.idx[i]]

            if mark[r] != c
                mark[r] = c
                p += one(I); targets(graph)[p] = r
            end

            if j != jprv
                r = cmp[S.idx[j]]

                if mark[r] != c
                    mark[r] = c
                    p += one(I); targets(graph)[p] = r
                end
            end

            jprv = j
        end
    end

    pointers(graph)[m + one(I)] = p + one(I)
    return trilinegraph(graph, reverse(graph))
end

struct Problem{T, I}
    G::SparseMatrixCSC{T, I}
    C::SparseMatrixCSC{T, I}
    A::SparseMatrixCSC{T, I}
    b::FVector{T}
    P::FPermutation{I}
    Q::FPermutation{I}
    k::I
    S::ChordalSymbolic{I}
    indices_primal::FVector{I}
    indices_slack::FVector{I}
    cgraph::FBipartiteGraph{I, I}
end

function Problem(
        C::SparseMatrixCSC{T},
        A::SparseMatrixCSC,
        b::AbstractVector{T},
        uplo::Char;
        alg = DEFAULT_ELIMINATION_ALGORITHM,
        tol = SPARSITY_THRESHOLD,
    ) where {T}
    G = pattern(C, A, uplo)
    P, S = symbolic(G; alg)

    Cp = sympermute(C, P.invp, uplo, 'L')
    Gp = sympermute(G, P.invp, uplo, 'L')
    Ap = sympermutepacked(A, P.invp, uplo, 'L')

    Q, k = permuteconstraints(Ap, tol)
    Ap = Ap / Q
    bp = FVector{T}(Q * b)

    indices_primal = compute_indices_primal(S, Ap)
    indices_slack = compute_indices_slack(Gp, Ap)
    cgraph = constraint_graph(S, Ap)

    return Problem(Gp, Cp, Ap, bp, P, Q, k, S, indices_primal, indices_slack, cgraph)
end

function Problem(
        C::AbstractMatrix,
        A::SparseMatrixCSC,
        b::AbstractVector;
        alg = DEFAULT_ELIMINATION_ALGORITHM,
        tol = SPARSITY_THRESHOLD,
    )
    P, tP = unwrapsym(C)
    return Problem(P, A, b, tP; alg, tol)
end

function Base.copy(problem::Problem)
    return Problem(
        problem.G,
        copy(problem.C),
        copy(problem.A),
        copy(problem.b),
        problem.P,
        problem.Q,
        problem.k,
        problem.S,
        problem.indices_primal,
        problem.indices_slack,
        problem.cgraph,
    )
end

function unwrapsym(A::HermOrSym)
    return parent(A), A.uplo
end

function unwrapsym(A::AbstractMatrix)
    if istriu(A)
        return A, 'U'
    elseif istril(A) || issymmetric(A)
        return A, 'L'
    else
        error()
    end
end

function compute_indices_primal(S::ChordalSymbolic{I}, A::SparseMatrixCSC{T, I}) where {T, I}
    n = convert(I, ncl(S))
    m = convert(I, ndz(S))
    P = FVector{I}(undef, nnz(A))

    @inbounds for c in axes(A, 2)
        plo = A.colptr[c]
        phi = A.colptr[c + 1] - one(I)

        for f in fronts(S)
            res = neighbors(S.res, f)
            sep = neighbors(S.sep, f)

            nn = eltypedegree(S.res, f)
            na = eltypedegree(S.sep, f)

            rlo = first(res)
            rhi = last(res)

            if ispositive(na)
                slo = first(sep)
                shi = last(sep)
            end

            jprv = zero(I)
            sloc = one(I)

            doff = S.Dptr[f] - one(I)
            loff = S.Lptr[f] - one(I)

            while plo ≤ phi
                i, j = cart(n, rowvals(A)[plo])
                j > rhi && break

                if j ≥ rlo
                    if j > jprv
                        jprv = j

                        sloc = one(I)
                        jloc = j - rlo

                        doff = S.Dptr[f] + jloc * nn
                        loff = S.Lptr[f] + jloc * na
                    end

                    if rlo ≤ i ≤ rhi
                        P[plo] = i - rlo + doff
                    elseif ispositive(na) && slo ≤ i ≤ shi
                        while sloc <= na && sep[sloc] < i
                            sloc += one(I)
                        end

                        if sloc <= na && sep[sloc] == i
                            P[plo] = sloc + loff + m - one(I)
                        end
                    end
                end

                plo += one(I)
            end
        end
    end

    return P
end

function compute_indices_slack(G::SparseMatrixCSC{T, I}, A::SparseMatrixCSC{T, I}) where {T, I}
    n = convert(I, size(G, 1))
    P = FVector{I}(undef, nnz(A))

    @inbounds for c in axes(A, 2)
        plo = A.colptr[c]
        phi = A.colptr[c + 1] - one(I)

        for k in axes(G, 2)
            qlo = G.colptr[k]
            qhi = G.colptr[k + 1] - one(I)
            q = qlo

            while plo ≤ phi
                i, j = cart(n, rowvals(A)[plo])
                j > k && break

                if j ≥ k
                    while q ≤ qhi && rowvals(G)[q] < i
                        q += one(I)
                    end

                    P[plo] = q
                end

                plo += one(I)
            end
        end
    end

    return P
end

# ===== Helper functions for Problem construction =====

# Coordinate conversion helpers
function flat(i::I, j::I, n::I) where I <: Integer
    return (j - one(I)) * n + i
end

function cart(n::I, m::I) where I <: Integer
    j, i = divrem(m - one(I), n)
    return i + one(I), j + one(I)
end

function intriangle(i::Integer, j::Integer, uplo::Char)
    return uplo == 'L' && i >= j || uplo == 'U' && i <= j
end

function pattern(C::SparseMatrixCSC{T, I}, A::SparseMatrixCSC{T, I}, uplo::Char) where {T, I}
    p = zero(I)
    n = convert(I, size(C, 1))

    rows = Vector{I}(undef, nnz(C) + nnz(A) + n)
    cols = Vector{I}(undef, nnz(C) + nnz(A) + n)

    for i in one(I):n
        p += one(I)
        rows[p] = i
        cols[p] = i
    end

    for j in axes(C, 2)
        for q in nzrange(C, j)
            i = rowvals(C)[q]

            if intriangle(i, j, uplo)
                p += one(I)
                rows[p] = i
                cols[p] = j
            end
        end
    end

    for f in rowvals(A)
        i, j = cart(n, f)

        if intriangle(i, j, uplo)
            p += one(I)
            rows[p] = i
            cols[p] = j
        end
    end

    resize!(rows, p)
    resize!(cols, p)
    return sparse(rows, cols, ones(T, p), n, n)
end

function sympermutepacked(
        A::SparseMatrixCSC{T, I},
        invp::AbstractVector,
        src::Char,
        tgt::Char,
    ) where {T, I}
    n = convert(I, isqrt(size(A, 1)))
    m = zero(I)

    colptr = zeros(I, n * n + 1)

    @inbounds for c in axes(A, 2)
        for p in nzrange(A, c)
            i, j = cart(n, rowvals(A)[p])

            src == 'L' && i < j && continue
            src == 'U' && i > j && continue

            pi = invp[i]
            pj = invp[j]

            if tgt == 'L'
                lo, hi = minmax(pi, pj)
            else
                hi, lo = minmax(pi, pj)
            end

            f = flat(hi, lo, n)

            if f < n * n
                colptr[f + two(I)] += one(I)
            end

            m += one(I)
        end
    end

    rowval = Vector{I}(undef, m)
    nzval = Vector{T}(undef, m)

    colptr[1] = m = one(I)

    @inbounds for f in axes(A, 1)
        colptr[f + 1] = m += colptr[f + 1]
    end

    @inbounds for c in axes(A, 2)
        for p in nzrange(A, c)
            i, j = cart(n, rowvals(A)[p])

            src == 'L' && i < j && continue
            src == 'U' && i > j && continue

            v = nonzeros(A)[p]
            pi = invp[i]
            pj = invp[j]

            if tgt == 'L'
                lo, hi = minmax(pi, pj)
            else
                hi, lo = minmax(pi, pj)
            end

            f = flat(hi, lo, n)
            q = colptr[f + 1]
            rowval[q] = c

            if (i > j) == (pi > pj)
                nzval[q] = conj(v)
            else
                nzval[q] = v
            end

            colptr[f + 1] = q + one(I)
        end
    end

    B = SparseMatrixCSC{T, I}(reverse(size(A))..., colptr, rowval, nzval)
    return copy(adjoint(B))
end

function permuteconstraints(A::SparseMatrixCSC{T, I}, threshold::Real) where {T, I}
    n = convert(I, isqrt(size(A, 1)))
    m = convert(I, size(A, 2))

    perm = FVector{I}(undef, m)
    mark = FVector{Bool}(undef, n)
    fill!(mark, false)

    k = zero(I)
    ns = zero(I)

    @inbounds for c in one(I):m
        cnt = zero(I)

        for p in nzrange(A, c)
            i, j = cart(n, rowvals(A)[p])

            if !mark[i]
                mark[i] = true
                cnt += one(I)
            end

            if !mark[j]
                mark[j] = true
                cnt += one(I)
            end
        end

        if cnt > n * threshold
            k += one(I)
            perm[k] = c
        else
            ns += one(I)
            perm[m - ns + one(I)] = c
        end

        for p in nzrange(A, c)
            i, j = cart(n, rowvals(A)[p])
            mark[i] = false
            mark[j] = false
        end
    end

    return Permutation(perm), k
end

# ===== Display =====

function show_problem(io::IO, problem::Problem, indent::Int)
    n = size(problem.C, 1)
    m = size(problem.A, 2)
    S = problem.S
    pad = " "^indent

    t1, t2, t3 = typemin(Int), typemin(Int), typemin(Int)
    b1, b2, b3 = typemax(Int), typemax(Int), typemax(Int)
    ncones = nfr(S)

    for j in fronts(S)
        d = eltypedegree(S.res, j) + eltypedegree(S.sep, j)

        if d > t1
            t3, t2, t1 = t2, t1, d
        elseif d > t2
            t3, t2 = t2, d
        elseif d > t3
            t3 = d
        end

        if d < b1
            b3, b2, b1 = b2, b1, d
        elseif d < b2
            b3, b2 = b2, d
        elseif d < b3
            b3 = d
        end
    end

    if ncones == 0
        dimstr = "()"
    elseif ncones == 1
        dimstr = "($t1)"
    elseif ncones == 2
        dimstr = "($t1, $b1)"
    elseif ncones == 3
        dimstr = "($t1, $t2, $b1)"
    elseif ncones == 4
        dimstr = "($t1, $t2, $b2, $b1)"
    elseif ncones == 5
        dimstr = "($t1, $t2, $t3, $b2, $b1)"
    elseif ncones == 6
        dimstr = "($t1, $t2, $t3, $b3, $b2, $b1)"
    else
        dimstr = "($t1, $t2, $t3, ..., $b3, $b2, $b1)"
    end

    println(io, pad, "(P)  min  ⟨C, X⟩               (D)  max  ⟨b, y⟩")
    println(io, pad, "     s.t. Aᵀ(X) = b                 s.t. A(y) + Z = C")
    println(io, pad, "          X ⪰ 0                          Z ⪰ 0")
    println(io)
    dimA = "$n × $n × $m"
    dimC = "$n × $n"
    @printf(io, "%sdim(A): %-20s  nnz(A): %d\n", pad, dimA, nnz(problem.A))
    @printf(io, "%sdim(C): %-20s  nnz(C): %d\n", pad, dimC, nnz(problem.C))
    println(io, pad, "dim(b): $m")
    println(io)
    println(io, pad, "chordal decomposition:")
    println(io, pad, "  cones: ", ncones)
    print(io, pad, "  sizes: ", dimstr)
    return
end

function Base.show(io::IO, ::MIME"text/plain", problem::Problem{T, J}) where {T, J}
    println(io, "Problem{$T, $J}:")
    show_problem(io, problem, 2)
end
