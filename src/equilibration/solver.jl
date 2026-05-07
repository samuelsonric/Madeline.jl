"""
    EquilibrationSolver{T, I}

Solver for computing parameters for Ruiz equilibration.
"""
struct EquilibrationSolver{T, I}
    equil::EquilibrationResult{T}
    problem::Problem{T, I}
end

function EquilibrationSolver(problem::Problem{T, I}) where {T, I}
    equil = EquilibrationResult{T}(size(problem.A, 2))
    return EquilibrationSolver(equil, problem)
end

"""
    equilibrate!(problem; iter_limit=10, minscale=1e-4, maxscale=1e4)

Re-scale a semidefinite program using Ruiz equilibration. Returns
a `EquilibrationResult` object, which can be used to un-scale the
solution.
"""
function equilibrate!(problem::Problem; kwargs...)
    solver = EquilibrationSolver(problem)
    result = solve!(solver; kwargs...)
    equilibrate!(problem, result)
    return result
end

function CommonSolve.init(::Type{EquilibrationSolver}, problem::Problem)
    return EquilibrationSolver(problem)
end

function CommonSolve.step!(
        solver::EquilibrationSolver{T, I};
        minscale::Real = 1e-4,
        maxscale::Real = 1e4,
    ) where {T, I}
    equil = solver.equil
    problem = solver.problem
    C = problem.C
    A = problem.A
    b = problem.b
    D = equil.dual
    E = equil.primal
    τ = equil.objective

    γ = zero(T)

    for i in axes(A, 2)
        d = D[i]

        for p in nzrange(A, i)
            γ = max(γ, d * abs(nonzeros(A)[p]))
        end
    end

    for p in eachindex(nonzeros(C))
        γ = max(γ, τ * abs(nonzeros(C)[p]))
    end

    if ispositive(γ)
        E = clamp(sqrt(E) / sqrt(γ), convert(T, minscale), convert(T, maxscale))
    end

    for i in axes(A, 2)
        η = τ * abs(b[i])

        for p in nzrange(A, i)
            η = max(η, E * abs(nonzeros(A)[p]))
        end

        if ispositive(η)
            D[i] = clamp(sqrt(D[i]) / sqrt(η), convert(T, minscale), convert(T, maxscale))
        end
    end

    ξ = zero(T)

    for p in eachindex(nonzeros(C))
        ξ = max(ξ, E * abs(nonzeros(C)[p]))
    end

    for i in axes(A, 2)
        ξ = max(ξ, D[i] * abs(b[i]))
    end

    if ispositive(ξ)
        τ = clamp(sqrt(τ) / sqrt(ξ), convert(T, minscale), convert(T, maxscale))
    end

    equil.primal = E
    equil.objective = τ
    return solver
end

function CommonSolve.solve!(solver::EquilibrationSolver; iter_limit::Integer=10, kwargs...)
    for _ in 1:iter_limit
        step!(solver; kwargs...)
    end

    return solver.equil
end
