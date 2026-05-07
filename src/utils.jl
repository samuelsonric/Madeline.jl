# flip a scaling flag
#
#     Val(true) ↔ Val(false)
#
function flip(::Val{SCALE}) where {SCALE}
    if SCALE
        return Val(false)
    else
        return Val(true)
    end
end

# compute the determinant of a 2 x 2 matrix:
#
#    | a b |
#    | c d |
#
function det2x2(a, b, c, d)
    bc = b * c
    return fma(a, d, -bc) - fma(b, c, -bc)
end

# solve for x:
#
#   Ax = b
#
function solve2x2!(A::AbstractMatrix, b::AbstractVector)
    A₁₁, A₂₁, A₁₂, A₂₂ = A
    b₁, b₂ = b
    det = det2x2(A₁₁, A₁₂, A₂₁, A₂₂)
    b[1] = det2x2(A₂₂, A₁₂, b₂, b₁) / det
    b[2] = det2x2(A₁₁, A₂₁, b₁, b₂) / det
    return b
end

# compute the norm of a symmetric matrix:
#
#     ‖A‖
#
function symnorm(A::SparseMatrixCSC{T}) where T
    nrm = zero(T)

    for j in axes(A, 2)
        for p in nzrange(A, j)
            i = rowvals(A)[p]
            v = nonzeros(A)[p]

            if i == j
                nrm += abs2(v)
            else
                nrm += 2 * abs2(v)
            end
        end
    end

    return sqrt(nrm)
end

# compute the inner product
#
#     ⟨X, Ac⟩
#
function dotpacked(
        X::ChordalTriangular{:N, :L, T, I},
        A::SparseMatrixCSC{T},
        indices::AbstractVector{I},
        c::Int,
    ) where {T, I}
    n = convert(I, isqrt(size(A, 1)))
    out = zero(T)

    @inbounds for p in nzrange(A, c)
        i, j = cart(n, rowvals(A)[p])

        Aij = nonzeros(A)[p]
        Xij = getflatindex(X, indices[p])

        out += Δ = Aij * Xij

        if i != j
            out += conj(Δ)
        end
    end

    return real(out)
end

# copy
#
#     Y ← Ac
#
function copytopacked!(Y::AbstractMatrix, A::SparseMatrixCSC, indices::AbstractVector, c::Integer)
    fill!(Y, false)
    return axpypacked!(true, A, indices, c, Y)
end

# copy
#
#     [ τ ] ← [-bc ]
#     [ X ]   [ Ac ]
#
function copytopacked!(w::Primal, A::SparseMatrixCSC, indices::AbstractVector, b::AbstractVector, c::Integer)
    copytopacked!(w.X, A, indices, c)
    w.τ = -b[c]
    return w
end

# compute the inner product
#
#     ⟨[ τ ], [-bc ]⟩
#     ⟨[ X ]  [ Ac ]⟩
#
function dotpacked(w::Primal, A::SparseMatrixCSC, indices::AbstractVector, b::AbstractVector, c::Integer)
    return dotpacked(w.X, A, indices, c) - w.τ * b[c]
end

# compute the sum-product
#
#     Y ← Y + α Ac
#
function axpypacked!(
        α::Number,
        A::SparseMatrixCSC{T},
        indices::AbstractVector{I},
        c::Int,
        Y::SparseMatrixCSC{T, I},
    ) where {T, I}
    @inbounds for p in nzrange(A, c)
        v = nonzeros(A)[p]
        idx = indices[p]
        nonzeros(Y)[idx] += α * v
    end

    return Y
end

# compute the sum-product
#
#     Y ← Y + α Ac
#
function axpypacked!(
        α::Number,
        A::SparseMatrixCSC{T},
        indices::AbstractVector{I},
        c::Int,
        Y::ChordalTriangular{:N, :L, T, I},
    ) where {T, I}
    @inbounds for p in nzrange(A, c)
        v = nonzeros(A)[p]
        idx = indices[p]
        setflatindex!(Y, getflatindex(Y, idx) + α * v, idx)
    end

    return Y
end

# compute the sum-product
#
#     [ τ ] ← [ τ ] + α [-bc ]
#     [ X ]   [ X ]     [ Ac ]
#
function axpypacked!(α::Number, A::SparseMatrixCSC, indices::AbstractVector, b::AbstractVector, c::Integer, w::Primal)
    axpypacked!(α, A, indices, c, w.X)
    w.τ -= α * b[c]
    return w
end

# compute the sum-product
#
#     y ← α Aᵀ X + β y
#
function apply_constraint_primal!(A::SparseMatrixCSC, indices::AbstractVector, X::AbstractMatrix, y::AbstractVector, α, β)
    if iszero(β)
        fill!(y, β)
    elseif !isone(β)
        rmul!(y, β)
    end

    for i in eachindex(y)
        y[i] += α * dotpacked(X, A, indices, i)
    end

    return y
end

# compute the sum-product
#
#     X ← α A y + β X
#
function apply_constraint_dual!(A::SparseMatrixCSC, indices::AbstractVector, X::AbstractMatrix, y::AbstractVector, α, β)
    if iszero(β)
        fill!(X, β)
    elseif !isone(β)
        rmul!(X, β)
    end

    for i in eachindex(y)
        axpypacked!(α * y[i], A, indices, i, X)
    end

    return X
end

# compute the sum-product
#
#     y ← α [-b Aᵀ ] [ τ ] + β y
#                    [ X ]
#
function apply_constraint_primal!(A::SparseMatrixCSC, indices::AbstractVector, b::AbstractVector, p::Point, y::AbstractVector, α, β)
    apply_constraint_primal!(A, indices, p.X, y, α, β)
    axpy!(-α * p.τ, b, y)
    return y
end

# compute the sum-product
#
#     [ κ ] ← α [-bᵀ] y + β [ κ ]
#     [ Z ]     [ A ]       [ Z ]
#
function apply_constraint_dual!(A::SparseMatrixCSC, indices::AbstractVector, b::AbstractVector, p::Point, y::AbstractVector, α, β)
    apply_constraint_dual!(A, indices, p.X, y, α, β)
    p.τ = β * p.τ - α * dot(b, y)
    return p
end

# compute the sum-product
#
# SCALE = true:
#
#     y ← α Aᵀ X + β y
#
# SCALE = false:
#
#     X ← α A y + β X
#
function apply_constraint!(A::SparseMatrixCSC, indices::AbstractVector, X::AbstractMatrix, y::AbstractVector, α, β, ::Val{SCALE}) where {SCALE}
    if SCALE
        return apply_constraint_primal!(A, indices, X, y, α, β)
    else
        return apply_constraint_dual!(A, indices, X, y, α, β)
    end
end

# compute the sum-product
#
# SCALE = true:
#
#     y ← α [-b Aᵀ ] [ τ ] + β y
#                    [ X ]
#
# SCALE = false:
#
#     [ κ ] ← α [-bᵀ] y + β [ κ ]
#     [ Z ]     [ A ]       [ Z ]
#
function apply_constraint!(A::SparseMatrixCSC, indices::AbstractVector, b::AbstractVector, p::Point, y::AbstractVector, α, β, ::Val{SCALE}) where {SCALE}
    if SCALE
        return apply_constraint_primal!(A, indices, b, p, y, α, β)
    else
        return apply_constraint_dual!(A, indices, b, p, y, α, β)
    end
end

# compute the sum-product
#
#     [ κ ] ← α [   -⟨C, ⋅⟩ ] [ τ ] + β [ κ ]
#     [ Z ]     [ C         ] [ X ]     [ Z ]
#
function apply_cost!(α, C::SparseMatrixCSC, x::Primal, β, out::Point)
    if iszero(β)
        out.τ = β
    elseif !isone(β)
        out.τ *= β
    end

    axpby_subset!(α * x.τ, C, β, out.X)
    out.τ -= α * symdot(x.X, C)
    return out
end

# compute the residual
#
#       y' ← -[-b Aᵀ ] [ τ ]
#                      [ X ]
#
#     [ κ' ] ← [   -⟨C, ⋅⟩ ] [ τ ] - [-bᵀ] y - [ κ ]
#     [ Z' ]   [ C         ] [ X ]   [ A ]     [ Z ]
#
function residual!(
        res::PrimalDualSlack{:L, T, J},
        itr::PrimalDualSlack{:L, T, J},
        problem::Problem{T, J},
    ) where {T, J}
    apply_constraint!(problem.A, problem.indices_primal, problem.b, itr.primal, res.dual, -one(T), zero(T), Val(true))
    apply_constraint!(problem.A, problem.indices_slack,  problem.b, res.slack,  itr.dual, -one(T), zero(T), Val(false))
    apply_cost!(one(T), problem.C, itr.primal, one(T), res.slack)
    axpy!(-one(T), itr.slack, res.slack)
    return res
end

# ===== Oracles =====

# compute the Hessian-vector product
#
# PRIMAL = true:
#
#     S ← ∇²f (U) S
#
# PRIMAL = false:
#
#     S ← ∇²f*(V) S
#
# where V = LLᵀ and
#
#     V = -∇f (U)
#     U = -∇f*(V)
#
function hessian!(
        space::Workspace{T, J},
        S::ChordalTriangular{:N, :L, T, J},
        L::ChordalTriangular{:N, :L, T, J},
        U::ChordalTriangular{:N, :L, T, J},
        primal::Val{PRIMAL},
    ) where {T, J, PRIMAL}
    fisher_impl!(space.Mptr, space.Mval, space.Fval, L, U, S, primal)
end

function hessianroot!(
        space::Workspace{T, J},
        S::ChordalTriangular{:N, :L, T, J},
        L::ChordalTriangular{:N, :L, T, J},
        U::ChordalTriangular{:N, :L, T, J},
        primal::Val{PRIMAL},
    ) where {T, J, PRIMAL}
    fisherroot_impl!(space.Mptr, space.Mval, space.Fval, L, U, S, primal, primal)
end

# compute the gradient
#
# PRIMAL = true:
#
#     [ κ ] ← ∇f ( [ τ ] )
#     [ Z ]      ( [ X ] )
#
# PRIMAL = false:
#
#     [ τ ] ← ∇f*( [ κ ] )
#     [ X ]      ( [ Z ] )
#
# where Z = LLᵀ.
#
function gradient!(
        space::Workspace{T, J},
        q::Primal{UPLO, T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        p::Point{UPLO, T, J},
        ::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    copyto!(q.X, L)

    if SCALE
        unchol_impl!(space.Mptr, space.Mval, space.Fval, space.Wval, q.X)
    else
        selinv_impl!(space.Mptr, space.Mval, space.Fval, q.X)
    end

    q.τ = inv(p.τ)
    return q
end

# compute the Hessian-vector product
#
# PRIMAL = true:
#
#     [ τ' ] ← ∇²f ( [ τ ] ) [ τ' ]
#     [ X' ]       ( [ X ] ) [ X' ]
#
# PRIMAL = false:
#
#     [ τ' ] ← ∇²f*( [ κ ] ) [ τ' ]
#     [ X' ]       ( [ Z ] ) [ X' ]
#
# where Z = LLᵀ and
#
#     [ κ ] = -∇f ( [ τ ] )
#     [ Z ]       ( [ X ] )
#
#     [ τ ] = -∇f*( [ κ ] )
#     [ X ]       ( [ Z ] )
#
function hessian!(
        space::Workspace{T, J},
        L::ChordalTriangular{:N, :L, T, J},
        x::Primal{:L, T, J},
        d::Primal{:L, T, J},
        primal::Val{PRIMAL},
    ) where {T, J, PRIMAL}
    if PRIMAL
        σ = inv(x.τ)^2
    else
        σ = x.τ^2
    end

    hessian!(space, d.X, L, x.X, primal)
    d.τ *= σ
    return d
end

function hessianroot!(
        space::Workspace{T, J},
        L::ChordalTriangular{:N, :L, T, J},
        x::Primal{:L, T, J},
        d::Primal{:L, T, J},
        primal::Val{PRIMAL},
    ) where {T, J, PRIMAL}
    if PRIMAL
        σ = inv(x.τ)
    else
        σ = x.τ
    end

    hessianroot!(space, d.X, L, x.X, primal)
    d.τ *= σ
    return d
end

function thirdorder!(
        space::Workspace{T, J},
        r::Primal{:L, T, J},
        L::ChordalTriangular{:N, :L, T, J},
        x::Primal{:L, T, J},
        d::Primal{:L, T, J},
        primal::Val{PRIMAL},
    ) where {T, J, PRIMAL}
    if PRIMAL
        σ = inv(x.τ)^3
    else
        σ = x.τ^3
    end

    amari_impl!(space.Mptr, space.Mval, space.Vval, space.Fval, r.X, d.X, d.X, L, x.X, primal)

    ldiv!(-two(T), r.X)
    r.τ = d.τ^2 * σ
    return r
end

function factorize!(
        space::Workspace{T, J},
        L::ChordalTriangular{:N, UPLO, T, J},
        p::Point{UPLO, T, J},
        ::Val{SCALE},
    ) where {UPLO, T, J, SCALE}
    copyto!(L, p.X)

    if SCALE
        info = complete_impl!(space.Mptr, space.Mval, space.Fval, L)
    else
        info = chol_impl!(space.Mptr, space.Mval, space.Fval, L)
    end

    return iszero(info)
end
