mutable struct Point{UPLO, T, I, Mat <: AbstractMatrix{T}}
    const X::Mat
    τ::T
end

################
# Point
################

function Base.fill!(h::Point, α)
    fill!(h.X, α)
    h.τ = α
    return h
end

function LinearAlgebra.ldiv!(α, h::Point)
    ldiv!(α, h.X)
    h.τ /= α
    return h
end

function LinearAlgebra.lmul!(α, h::Point)
    lmul!(α, h.X)
    h.τ *= α
    return h
end

function LinearAlgebra.norm(h::Point)
    return sqrt(dot(h, h))
end

################
# Primal
################

const Primal{UPLO, T, I} = Point{UPLO, T, I, FChordalTriangular{:N, UPLO, T, I}}

function Primal{UPLO, T}(S::ChordalSymbolic{I}) where {UPLO, T, I}
    X = FChordalTriangular{:N, UPLO, T, I}(S)
    return Primal{UPLO, T, I}(X, zero(T))
end

function Base.copyto!(dst::Primal, src::Primal)
    copyto!(dst.X, src.X)
    dst.τ = src.τ
    return dst
end

function LinearAlgebra.axpy!(α, x::Primal, y::Primal)
    axpy!(α, x.X, y.X)
    y.τ += α * x.τ
    return y
end

function LinearAlgebra.axpby!(α, x::Primal, β, y::Primal)
    axpby!(α, x.X, β, y.X)
    y.τ = α * x.τ + β * y.τ
    return y
end

function LinearAlgebra.dot(s::Primal, z::Primal)
    return symdot(s.X, z.X) + s.τ * z.τ
end

################
# Slack
################

const Slack{UPLO, T, I} = Point{UPLO, T, I, SparseMatrixCSC{T, I}}

function Slack{UPLO, T}(G::SparseMatrixCSC{<:Any, I}) where {UPLO, T, I}
    X = SparseMatrixCSC{T, I}(size(G)..., copy(G.colptr), copy(G.rowval), zeros(T, nnz(G)))
    return Slack{UPLO, T, I}(X, zero(T))
end

function Base.copyto!(dst::Slack, src::Slack)
    copyto!(nonzeros(dst.X), nonzeros(src.X))
    dst.τ = src.τ
    return dst
end

function LinearAlgebra.axpy!(α, x::Slack, y::Slack)
    axpy_sparse!(α, x.X, y.X)
    y.τ += α * x.τ
    return y
end

function LinearAlgebra.dot(s::Slack, z::Slack)
    return symdot_sparse(s.X, z.X) + s.τ * z.τ
end

################
# Cross-Type
################

function Base.copyto!(dst::Primal, src::Slack)
    copyto!(dst.X, src.X)
    dst.τ = src.τ
    return dst
end

function LinearAlgebra.axpy!(α, x::Slack, y::Primal)
    axpy!(α, x.X, y.X)
    y.τ += α * x.τ
    return y
end

function LinearAlgebra.dot(s::Primal, z::Slack)
    return symdot(s.X, z.X) + s.τ * z.τ
end

function LinearAlgebra.dot(s::Slack, z::Primal)
    return dot(z, s)
end

################
# Sparse Helpers
################

function axpy_sparse!(α, X::SparseMatrixCSC, Y::SparseMatrixCSC)
    axpy!(α, nonzeros(X), nonzeros(Y))
    return Y
end

function axpby_sparse!(α, X::SparseMatrixCSC, β, Y::SparseMatrixCSC)
    axpby!(α, nonzeros(X), β, nonzeros(Y))
    return Y
end

function axpy_subset!(α, X::SparseMatrixCSC{T, I}, Y::SparseMatrixCSC{T, I}) where {T, I}
    @inbounds for k in axes(X, 2)
        px   = X.colptr[k]
        pxhi = X.colptr[k + one(I)] - one(I)

        py   = Y.colptr[k]
        pyhi = Y.colptr[k + one(I)] - one(I)

        while px <= pxhi
            i = rowvals(X)[px]

            while py <= pyhi && rowvals(Y)[py] < i
                py += one(I)
            end

            if py <= pyhi && rowvals(Y)[py] == i
                nonzeros(Y)[py] += α * nonzeros(X)[px]
            end

            px += one(I)
        end
    end

    return Y
end

function axpby_subset!(α, X::SparseMatrixCSC{T, I}, β, Y::SparseMatrixCSC{T, I}) where {T, I}
    if iszero(β)
        fill!(Y, β)
    elseif !isone(β)
        rmul!(Y, β)
    end

    return axpy_subset!(α, X, Y)
end

function axpby_subset!(α, X::SparseMatrixCSC, β, Y::ChordalTriangular)
    return axpby!(α, X, β, Y)
end

function symdot_sparse(A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T}) where {T}
    out = zero(T)

    @inbounds for j in axes(A, 2)
        for p in nzrange(A, j)
            out += Δ = nonzeros(A)[p] * nonzeros(B)[p]

            if rowvals(A)[p] != j
                out += conj(Δ)
            end
        end
    end

    return out
end

function copyto_sparse!(dst::SparseMatrixCSC, src::SparseMatrixCSC)
    copyto!(nonzeros(dst), nonzeros(src))
    return dst
end
