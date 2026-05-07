mutable struct State{T}
    const ncol::Int
    nitr::Int
    nslw::Int
    pobj::T
    dobj::T
    pres::T
    dres::T
    pinf::T
    dinf::T
    prox::T
    τ::T
    κ::T
    μ::T
    status::Status
end

function State{T}(n::Int) where {T}
    return State{T}(
        n, 0, 0,
        typemax(T), typemax(T), typemax(T),
        typemax(T), typemax(T), typemax(T),
        typemax(T),
        one(T), one(T), one(T),
        CONTINUE,
    )
end


# Derived quantities (computed on the fly from μ, τ, κ, n)
function gap(s::State)
    return s.μ * (s.ncol + 1) - s.τ * s.κ
end

function relgap(s::State{T}) where {T}
    g = gap(s)
    if isnegative(s.pobj)
        return g / (-s.pobj * s.τ)
    elseif ispositive(s.dobj)
        return g / (s.dobj * s.τ)
    else
        return typemax(T)
    end
end

function Base.copyto!(dst::State, src::State)
    dst.status = src.status
    dst.nitr = src.nitr
    dst.pobj = src.pobj
    dst.dobj = src.dobj
    dst.pres = src.pres
    dst.dres = src.dres
    dst.pinf = src.pinf
    dst.dinf = src.dinf
    dst.τ = src.τ
    dst.κ = src.κ
    dst.μ = src.μ
    dst.prox = src.prox
    dst.nslw = src.nslw
    return dst
end

function print_header()
    @printf("%5s %12s %12s |%9s %9s %9s |%9s %9s |%8s\n",
            "iter", "pobj", "dobj", "gap", "pres", "dres", "κ/τ", "μ", "prox")
    println("-"^93)
    return
end

function print_loop(state::State)
    τ = state.τ
    pobj = state.pobj / τ
    dobj = state.dobj / τ
    g = gap(state) / τ^2
    pres = state.pres / τ
    dres = state.dres / τ
    κτ = state.κ / τ

    if iszero(state.nitr)
        @printf("%5d %12.4e %12.4e |%9.2e %9.2e %9.2e |%9.2e %9.2e |\n",
                state.nitr, pobj, dobj, g, pres, dres, κτ, state.μ)
    else
        @printf("%5d %12.4e %12.4e |%9.2e %9.2e %9.2e |%9.2e %9.2e |%8.1e\n",
                state.nitr, pobj, dobj, g, pres, dres, κτ, state.μ, state.prox)
    end

    return
end

function show_state(io::IO, state::State, indent::Int)
    τ = state.τ
    pobj = state.pobj / τ
    dobj = state.dobj / τ
    g = gap(state) / τ^2
    pres = state.pres / τ
    dres = state.dres / τ
    κτ = state.κ / τ
    pad = " "^indent

    @printf(io, "%sstatus: %-12s  iter: %d\n", pad, state.status, state.nitr)
    @printf(io, "%spobj: %12.4e    dobj: %12.4e\n", pad, pobj, dobj)
    @printf(io, "%spres: %12.2e    dres: %12.2e\n", pad, pres, dres)
    @printf(io, "%sgap:  %12.2e    prox: %12.2e\n", pad, g, state.prox)
    @printf(io, "%sμ:    %12.2e    κ/τ:  %12.2e", pad, state.μ, κτ)
    return
end

function Base.show(io::IO, ::MIME"text/plain", state::State{T}) where {T}
    println(io, "State{$T}:")
    show_state(io, state, 2)
end

function print_terminated(state::State)
    println("-"^93)
    println("Terminated with status ", state.status, ".")
    return
end
