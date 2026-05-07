mutable struct EquilibrationResult{T}
    primal::T
    objective::T
    const dual::FVector{T}
end

function EquilibrationResult{T}(n::Integer) where {T}
    dual = FVector{T}(undef, n)
    return EquilibrationResult(one(T), one(T), fill!(dual, one(T)))
end

"""
    equilibrate!(problem::Problem, equil::EquilibrationResult)

Re-scale a semidefinite program using the paremeters in `equil`.
The solution must must be un-scaled using [`deequilibrate!`](@ref).
"""
function equilibrate!(problem::Problem, equil::EquilibrationResult)
    C = problem.C
    A = problem.A
    b = problem.b

    D = equil.dual
    E = equil.primal
    τ = equil.objective

    for i in axes(A, 2)
        b[i] *= d = D[i]

        for p in nzrange(A, i)
            nonzeros(A)[p] *= d
        end
    end

    lmul!(    τ, b)
    lmul!(E,     A)
    lmul!(E * τ, C)

    return problem
end

"""
    deequilibrate!(result, equil)
"""
function deequilibrate!(result::Result, equil::EquilibrationResult)
    τ = equil.objective
    lmul!(equil.primal / τ, result.itr.primal.X)
    result.itr.dual .*= equil.dual
    result.itr.dual ./= τ
    result.state.pobj /= τ^2
    result.state.dobj /= τ^2
    result.state.μ /= τ^2
    result.state.κ /= τ^2
    return result
end
