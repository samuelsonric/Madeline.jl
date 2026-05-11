# History for slow progress detection
struct History{T}
    n::Int
    pres::Vector{T}
    dres::Vector{T}
    τ::Vector{T}
    μ::Vector{T}
end

function History{T}(n::Int) where T
    return History{T}(
        n,
        fill(typemax(T), n),
        fill(typemax(T), n),
        fill(one(T), n),
        fill(one(T), n),
    )
end

function Base.push!(h::History, s::State)
    i = mod1(s.nitr, h.n)
    h.pres[i] = s.pres
    h.dres[i] = s.dres
    h.τ[i] = tau(s)
    h.μ[i] = s.μ
    return h
end

function max_abs_diff(h::History{T}) where T
    val = zero(T)
    for i in 1:h.n
        j = mod1(i - 1, h.n)
        c1 = max(h.pres[i], h.dres[i]) / h.τ[i]
        c2 = max(h.pres[j], h.dres[j]) / h.τ[j]
        val = max(val, abs(c1 - c2))
    end
    return val
end

function firstscore(h::History{T}, nitr::Int) where T
    i = mod1(nitr + 1, h.n)
    return max(h.pres[i], h.dres[i]) / h.τ[i]
end
