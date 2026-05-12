# ===== Problem data type =====

const SPARSITY_THRESHOLD_SCHUR = 0.75

# Count connected components (trees) in the elimination forest.
function count_connected_components(S::ChordalSymbolic{I}) where {I}
    ncc = zero(I)

    for i in fronts(S)
        if iszero(S.pnt[i])
            ncc += one(I)
        end
    end

    return ncc
end

# Compute connected components of the elimination forest.
# Returns (frnt_to_cc, frtptr, ncc) where:
#   - frnt_to_cc[f] = CC index for front f (increasing with front order)
#   - frtptr[c]:frtptr[c+1]-1 = front range for CC c
#   - ncc = number of connected components
function connected_components(S::ChordalSymbolic{I}) where {I}
    nf = convert(I, nfr(S))
    ncc = count_connected_components(S)

    frnt_to_cc = FVector{I}(undef, nf)
    frtptr = FVector{I}(undef, ncc + one(I))

    # Reverse pass: fill frnt_to_cc (countdown so CCs increase with fronts)
    c = ncc + one(I)

    for i in reverse(fronts(S))
        j = S.pnt[i]

        if iszero(j)
            frnt_to_cc[i] = c -= one(I)
        else
            frnt_to_cc[i] = frnt_to_cc[j]
        end
    end

    # Forward pass: build frtptr
    cprev = zero(I)

    for i in fronts(S)
        c = frnt_to_cc[i]

        while cprev < c
            cprev += one(I)
            frtptr[cprev] = i
        end
    end

    frtptr[ncc + one(I)] = nf + one(I)

    return frnt_to_cc, frtptr, ncc
end

# Compute touched rows for sparse constraints (k+1:m).
# Returns (idxfwd, idxbwd, idxptr, nrhs) where:
#   - idxfwd[1:nrhs] = global row indices of touched rows
#   - idxbwd[row] = local index in idxfwd (0 if not touched)
#   - idxptr[c]:idxptr[c+1]-1 = range in idxfwd for CC c
#   - nrhs = number of touched rows
function touched(
        A::SparseMatrixCSC{T, I},
        k::I,
        S::ChordalSymbolic{I},
        frnt_to_cc::FVector{I},
        ncc::I,
    ) where {T, I}
    n = convert(I, ncl(S))
    m = convert(I, size(A, 2))

    idxfwd = FVector{I}(undef, n)
    idxbwd = FVector{I}(undef, n)
    idxptr = FVector{I}(undef, ncc + one(I))

    fill!(idxfwd, zero(I))
    fill!(idxbwd, zero(I))

    @inbounds for c in k + one(I):m
        for p in nzrange(A, c)
            i, j = cart(n, rowvals(A)[p])
            idxbwd[i] = one(I)
            idxbwd[j] = one(I)
        end
    end

    nrhs = ccold = zero(I)

    @inbounds for i in oneto(n)
        if !iszero(idxbwd[i])
            idxbwd[i] = nrhs += one(I)
            idxfwd[nrhs] = i

            cc = frnt_to_cc[S.idx[i]]

            while ccold < cc
                ccold += one(I); idxptr[ccold] = nrhs
            end
        end
    end

    while ccold ≤ ncc
        ccold += one(I); idxptr[ccold] = nrhs + one(I)
    end

    return idxfwd, idxbwd, idxptr, nrhs
end

function constraint_graph(S::ChordalSymbolic{I}, A::SparseMatrixCSC{T, I}) where {T, I}
    m = convert(I, size(A, 2))
    n = convert(I, ncl(S))

    frnt_to_cc, frtptr, ncc = connected_components(S)

    p = zero(I)

    for c in oneto(m)
        rprv = zero(I)

        for q in nzrange(A, c)
            i, j = cart(n, rowvals(A)[q])

            r = frnt_to_cc[S.idx[j]]

            if r != rprv
                p += one(I)
            end

            rprv = r
        end
    end

    cons_to_cc = FBipartiteGraph{I, I}(ncc, m, p)
    cc_to_strt = FBipartiteGraph{I, I}(nnz(A), ncc, p)
    cc_to_stop = FBipartiteGraph{I, I}(nnz(A), ncc, p)

    p = zero(I)

    for c in oneto(m)
        pointers(cons_to_cc)[c] = p + one(I)

        rprv = zero(I)

        for q in nzrange(A, c)
            i, j = cart(n, rowvals(A)[q])

            r = frnt_to_cc[S.idx[j]]

            if r != rprv
                p += one(I); targets(cons_to_cc)[p] = r
            end

            rprv = r
        end
    end

    pointers(cons_to_cc)[m + one(I)] = p + one(I)

    cc_to_cons = reverse(cons_to_cc)

    mark = FVector{I}(undef, m)

    for c in oneto(m)
        mark[c] = A.colptr[c]
    end

    p = zero(I)

    for cc in vertices(cc_to_cons)
        pointers(cc_to_strt)[cc] = p + one(I)
        pointers(cc_to_stop)[cc] = p + one(I)

        for c in neighbors(cc_to_cons, cc)
            p += one(I)

            qstrt = q = mark[c]
            qstop = A.colptr[c + one(I)]

            while q < qstop
                i, j = cart(n, rowvals(A)[q])
                r = frnt_to_cc[S.idx[j]]

                if r != cc
                    mark[c] = qstop = q
                end

                q += one(I)
            end

            targets(cc_to_strt)[p] = qstrt
            targets(cc_to_stop)[p] = qstop - one(I)
        end
    end

    pointers(cc_to_strt)[ncc + one(I)] = p + one(I)
    pointers(cc_to_stop)[ncc + one(I)] = p + one(I)

    return cons_to_cc, cc_to_cons, cc_to_strt, cc_to_stop, frnt_to_cc, frtptr, ncc
end

struct Problem{T, I, DualPat, DualPerm, DualSymb}
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
    cons_to_cc::FBipartiteGraph{I, I}
    cc_to_cons::FBipartiteGraph{I, I}
    cc_to_strt::FBipartiteGraph{I, I}
    cc_to_stop::FBipartiteGraph{I, I}
    frnt_to_cc::FVector{I}
    frtptr::FVector{I}
    ncc::I
    idxfwd::FVector{I}
    idxbwd::FVector{I}
    idxptr::FVector{I}
    nrhs::I
    # Workspace sizing for OffsetArray optimization
    max_rhs_per_cc::I
    max_cc_rows::I
    max_cons_per_cc::I
    # Dual (Schur complement) sparsity pattern and symbolic factorization
    dual_patt::DualPat
    dual_perm::DualPerm
    dual_symb::DualSymb
end

const DenseProblem{T, I} = Problem{T, I, Nothing, Nothing, Nothing}
const SparseProblem{T, I} = Problem{T, I, SparseMatrixCSC{T, I}, FPermutation{I}, ChordalSymbolic{I}}

function linegraph_sparsity(ve::FBipartiteGraph{I, I}, ev::FBipartiteGraph{I, I}) where {I}
    n = nv(ve)
    m = zero(I)
    marker = FVector{I}(undef, n)
    fill!(marker, zero(I))

    @inbounds for v in vertices(ve)
        for w in neighbors(ve, v), x in neighbors(ev, w)
            if x > v && marker[x] < v
                marker[x] = v
                m += one(I)
            end
        end
    end

    return 1 - (2m + n) / (n * n)
end

function Problem(
        C::SparseMatrixCSC{T},
        A::SparseMatrixCSC,
        b::AbstractVector{T},
        uplo::Char;
        alg = DEFAULT_ELIMINATION_ALGORITHM,
        tol = SPARSITY_THRESHOLD,
    ) where {T}
    @timeit TIMER "pattern" G = pattern(C, A, uplo)
    @timeit TIMER "primal_symbolic" P, S = symbolic(G; alg)

    @timeit TIMER "sympermute_C" Cp = sympermute(C, P.invp, uplo, 'L')
    @timeit TIMER "sympermute_G" Gp = sympermute(G, P.invp, uplo, 'L')
    @timeit TIMER "sympermute_A" Ap = sympermutepacked(A, P.invp, uplo, 'L')

    @timeit TIMER "permute_constraints" Q, k = permuteconstraints(Ap, tol)
    Ap = Ap / Q
    bp = FVector{T}(Q * b)

    @timeit TIMER "indices_primal" indices_primal = compute_indices_primal(S, Ap)
    @timeit TIMER "indices_slack" indices_slack = compute_indices_slack(Gp, Ap)
    @timeit TIMER "constraint_graph" cons_to_cc, cc_to_cons, cc_to_strt, cc_to_stop, frnt_to_cc, frtptr, ncc = constraint_graph(S, Ap)
    @timeit TIMER "touched" idxfwd, idxbwd, idxptr, nrhs = touched(Ap, k, S, frnt_to_cc, ncc)

    # Compute workspace sizes for OffsetArray optimization
    I = typeof(ncc)
    max_rhs_per_cc = zero(I)
    for c in oneto(ncc)
        rhs_count = idxptr[c + one(I)] - idxptr[c]
        max_rhs_per_cc = max(max_rhs_per_cc, rhs_count)
    end

    res_ptr = S.res.ptr
    max_cc_rows = zero(I)

    for c in oneto(ncc)
        fdsc = frtptr[c]
        root = frtptr[c + one(I)] - one(I)
        row_lo = res_ptr[fdsc]
        row_hi = res_ptr[root + one(I)] - one(I)
        max_cc_rows = max(max_cc_rows, row_hi - row_lo + one(I))
    end

    max_cons_per_cc = zero(I)
    for c in oneto(ncc)
        cons_count = pointers(cc_to_cons)[c + one(I)] - pointers(cc_to_cons)[c]
        max_cons_per_cc = max(max_cons_per_cc, cons_count)
    end

    @timeit TIMER "linegraph_sparsity" schur_sparsity = T(linegraph_sparsity(cons_to_cc, cc_to_cons))

    # Compute dual (Schur complement) symbolic factorization if sparse
    if schur_sparsity > SPARSITY_THRESHOLD_SCHUR
        @timeit TIMER "linegraph" LG = linegraph(cons_to_cc, cc_to_cons)
        @timeit TIMER "dual_patt" dual_patt = sparse(T, I, LG)
        @timeit TIMER "dual_symbolic" dual_perm, dual_symb = symbolic(dual_patt; alg)
    else
        dual_patt = nothing
        dual_perm = nothing
        dual_symb = nothing
    end

    return Problem(Gp, Cp, Ap, bp, P, Q, k, S, indices_primal, indices_slack, cons_to_cc, cc_to_cons, cc_to_strt, cc_to_stop, frnt_to_cc, frtptr, ncc, idxfwd, idxbwd, idxptr, nrhs, max_rhs_per_cc, max_cc_rows, max_cons_per_cc, dual_patt, dual_perm, dual_symb)
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
        problem.cons_to_cc,
        problem.cc_to_cons,
        problem.cc_to_strt,
        problem.cc_to_stop,
        problem.frnt_to_cc,
        problem.frtptr,
        problem.ncc,
        problem.idxfwd,
        problem.idxbwd,
        problem.idxptr,
        problem.nrhs,
        problem.max_rhs_per_cc,
        problem.max_cc_rows,
        problem.max_cons_per_cc,
        problem.dual_patt,
        problem.dual_perm,
        problem.dual_symb,
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

    local f, res, sep, nn, na, rlo, slo, shi, Dp, Lp

    @inbounds for c in axes(A, 2)
        jprv = rhi = zero(I)

        for p in nzrange(A, c)
            i, j = cart(n, rowvals(A)[p])

            if rhi < j
                f = S.idx[j]

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

                Dp = S.Dptr[f]
                Lp = S.Lptr[f]
            end
    
            rloc = i - rlo

            if jprv < j
                jprv = j
                sloc = one(I)
                jloc = j - rlo
            end

            if rlo ≤ i ≤ rhi
                P[p] = rloc + Dp + jloc * nn
            elseif ispositive(na) && slo ≤ i ≤ shi
                while sloc <= na && sep[sloc] < i
                    sloc += one(I)
                end

                if sloc <= na && sep[sloc] == i
                    P[p] = sloc + Lp + jloc * na + m - one(I)
                end
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

function sympermutepacked_loop!(
        work::AbstractVector{Tuple{I, T}},
        A::SparseMatrixCSC{T, I},
        invp::AbstractVector{I},
        src::Char,
        tgt::Char,
        c::I,
    ) where {T, I}
    n = convert(I, isqrt(size(A, 1)))
    empty!(work)

    @inbounds for p in nzrange(A, c)
        i, j = cart(n, rowvals(A)[p])

        src == 'L' && i < j && continue
        src == 'U' && i > j && continue

        v = nonzeros(A)[p]
        pi = invp[i]
        pj = invp[j]

        if tgt == 'L'
            pj, pi = minmax(pi, pj)
        else
            pi, pj = minmax(pi, pj)
        end

        pp = flat(pi, pj, n)

        push!(work, (pp, v))
    end

    sort!(work; by=first, alg=QuickSort)
    return
end

function sympermutepacked(
        A::SparseMatrixCSC{T, I},
        invp::AbstractVector{I},
        src::Char,
        tgt::Char,
    ) where {T, I}
    n = convert(I, isqrt(size(A, 1)))
    m = convert(I, size(A, 2))
    k = convert(I, nnz(A))

    work = Tuple{I, T}[]
    colptr = Vector{I}(undef, m + one(I))
    rowval = Vector{I}(undef, k)
    nzval = Vector{T}(undef, k)

    p = zero(I)

    @inbounds for c in oneto(m)
        colptr[c] = p + one(I)
        sympermutepacked_loop!(work, A, invp, src, tgt, c)

        for (i, v) in work
            p += one(I); rowval[p] = i
                          nzval[p] = v
        end
    end

    colptr[m + one(I)] = p + one(I)
    resize!(rowval, p)
    resize!(nzval, p)
    return SparseMatrixCSC{T, I}(n * n, m, colptr, rowval, nzval)
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

    println(io, pad, "  (P)  min  ⟨C, X⟩               (D)  max  ⟨b, y⟩")
    println(io, pad, "       s.t. Aᵀ(X) = b                 s.t. A(y) + Z = C")
    println(io, pad, "            X ⪰ 0                          Z ⪰ 0")
    println(io)
    println(io, pad, "problem:")
    dimA = "$n × $n × $m"
    dimC = "$n × $n"
    @printf(io, "%s  dim(A): %-21s  nnz(A): %d\n", pad, dimA, nnz(problem.A))
    @printf(io, "%s  dim(C): %-21s  nnz(C): %d\n", pad, dimC, nnz(problem.C))
    println(io, pad, "  dim(b): $m")
    println(io)

    # Aggregate sparsity (primal Z)
    println(io, pad, "aggregate sparsity:")
    nnz_Z = nnz(problem.G)
    lnz_Z = nnz(S)
    dimZ = "$n × $n"
    @printf(io, "%s  dim(Z): %-21s  nnz(Z): %d\n", pad, dimZ, nnz_Z)
    @printf(io, "%s                                 lnz(Z): %d\n", pad, lnz_Z)
    println(io)

    # Correlative sparsity (dual Schur complement)
    println(io, pad, "correlative sparsity:")
    dimS = "$m × $m"
    if problem isa DenseProblem
        nnz_S = m * (m + 1) ÷ 2
        @printf(io, "%s  dim(S): %-21s  nnz(S): %d  (dense)\n", pad, dimS, nnz_S)
    else
        nnz_S = nnz(problem.dual_patt)
        lnz_S = nnz(problem.dual_symb)
        @printf(io, "%s  dim(S): %-21s  nnz(S): %d\n", pad, dimS, nnz_S)
        @printf(io, "%s                                 lnz(S): %d\n", pad, lnz_S)
    end
    return
end

function Base.show(io::IO, ::MIME"text/plain", problem::Problem{T, J}) where {T, J}
    println(io, "Problem{$T, $J}:")
    show_problem(io, problem, 2)
end
