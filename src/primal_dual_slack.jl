struct PrimalDualSlack{UPLO, T, I}
    dual::FVector{T}
    primal::Primal{UPLO, T, I}
    slack::Slack{UPLO, T, I}
end

function PrimalDualSlack{UPLO, T}(m::Integer, S::ChordalSymbolic, G::SparseMatrixCSC) where {UPLO, T}
    dual = FVector{T}(undef, m)
    primal = Primal{UPLO, T}(S)
    slack = Slack{UPLO, T}(G)
    return PrimalDualSlack(dual, primal, slack)
end

function Base.copyto!(dst::PrimalDualSlack, src::PrimalDualSlack)
    copyto!(dst.primal, src.primal)
    copyto!(dst.dual, src.dual)
    copyto!(dst.slack, src.slack)
    return dst
end

function LinearAlgebra.axpy!(α, x::PrimalDualSlack, y::PrimalDualSlack)
    axpy!(α, x.primal, y.primal)
    axpy!(α, x.dual, y.dual)
    axpy!(α, x.slack, y.slack)
    return y
end

function Base.fill!(p::PrimalDualSlack, α)
    fill!(p.primal, α)
    fill!(p.dual, α)
    fill!(p.slack, α)
    return p
end

function LinearAlgebra.ldiv!(α, p::PrimalDualSlack)
    ldiv!(α, p.primal)
    ldiv!(α, p.dual)
    ldiv!(α, p.slack)
    return p
end

function LinearAlgebra.norm(p::PrimalDualSlack)
    return hypot(norm(p.dual), norm(p.primal), norm(p.slack))
end
