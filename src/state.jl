mutable struct State{UPLO, T, I}
    nitr::Int
    nslw::Int
    pobj::T
    dobj::T
    pres::T
    dres::T
    pinf::T
    dinf::T
    prox::T
    μ::T
    status::Status
    const itr::PrimalDualSlack{UPLO, T, I}
end

# Accessors for values stored in itr
tau(s::State) = s.itr.primal.τ
kap(s::State) = s.itr.slack.τ
Multifrontal.ncl(s::State) = size(s.itr.primal.X, 1)

function State{UPLO, T}(m::Integer, S::ChordalSymbolic{I}, G::SparseMatrixCSC) where {UPLO, T, I}
    itr = PrimalDualSlack{UPLO, T}(m, S, G)
    return State{UPLO, T, I}(
        0, 0,
        typemax(T), typemax(T), typemax(T),
        typemax(T), typemax(T), typemax(T),
        typemax(T),
        one(T),
        CONTINUE,
        itr,
    )
end


# Derived quantities
function gap(s::State)
    return s.μ * (ncl(s) + 1) - tau(s) * kap(s)
end

# Score for best iterate comparison (lower is better)
score(s::State) = max(s.pres, s.dres) / tau(s)

function relgap(s::State{UPLO, T, I}) where {UPLO, T, I}
    g = gap(s)
    τ = tau(s)
    if isnegative(s.pobj)
        return g / (-s.pobj * τ)
    elseif ispositive(s.dobj)
        return g / (s.dobj * τ)
    else
        return typemax(T)
    end
end

function Base.copyto!(dst::State, src::State)
    # Note: status is NOT copied
    dst.nitr = src.nitr
    dst.nslw = src.nslw
    dst.pobj = src.pobj
    dst.dobj = src.dobj
    dst.pres = src.pres
    dst.dres = src.dres
    dst.pinf = src.pinf
    dst.dinf = src.dinf
    dst.prox = src.prox
    dst.μ = src.μ
    copyto!(dst.itr, src.itr)
    return dst
end

function print_header()
    @printf("%5s %12s %12s |%9s %9s %9s |%9s %9s |%8s\n",
            "iter", "pobj", "dobj", "gap", "pres", "dres", "κ/τ", "μ", "prox")
    println("-"^93)
    return
end

function print_loop(state::State)
    τ = tau(state)
    pobj = state.pobj / τ
    dobj = state.dobj / τ
    g = gap(state) / τ^2
    pres = state.pres / τ
    dres = state.dres / τ
    κτ = kap(state) / τ

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
    τ = tau(state)
    pobj = state.pobj / τ
    dobj = state.dobj / τ
    g = gap(state) / τ^2
    pres = state.pres / τ
    dres = state.dres / τ
    κτ = kap(state) / τ
    pad = " "^indent

    @printf(io, "%sstatus: %-12s  iter: %d\n", pad, state.status, state.nitr)
    @printf(io, "%spobj: %12.4e    dobj: %12.4e\n", pad, pobj, dobj)
    @printf(io, "%spres: %12.2e    dres: %12.2e\n", pad, pres, dres)
    @printf(io, "%sgap:  %12.2e    prox: %12.2e\n", pad, g, state.prox)
    @printf(io, "%sμ:    %12.2e    κ/τ:  %12.2e", pad, state.μ, κτ)
    return
end

function Base.show(io::IO, ::MIME"text/plain", state::State{UPLO, T, I}) where {UPLO, T, I}
    println(io, "State{:$UPLO, $T, $I}:")
    show_state(io, state, 2)
end

function print_restored(state::State)
    println("Restored from iteration $(state.nitr).")
    return
end

function print_terminated(state::State)
    println("-"^93)
    println("Terminated with status ", state.status, ".")
    return
end
