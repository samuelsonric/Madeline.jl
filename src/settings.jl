@kwdef struct Settings{T}
    rel_opt::T      = sqrt(eps(T))
    abs_opt::T      = eps(T)^T(0.75)
    feas::T         = sqrt(eps(T))
    infeas::T       = eps(T)^T(0.75)
    tau_infeas::T   = T(1e-2)
    illposed::T     = eps(T)^T(0.75)
    slow::T         = T(1e-3)
    near_factor::T  = T(1000)
    iter_limit::Int = 1000
    prox_bound::T   = T(0.99)
    scaling::Bool   = true
    verbose::Bool   = true
end

function show_settings(io::IO, settings::Settings, indent::Int)
    pad = " "^indent
    @printf(io, "%sscaling:       %s\n", pad, settings.scaling ? "primal" : "dual")
    @printf(io, "%siter_limit:    %-7d     prox_bound:    %.2e\n", pad, settings.iter_limit, settings.prox_bound)
    @printf(io, "%stol_rel_opt:   %.1e     tol_abs_opt:   %.1e\n", pad, settings.rel_opt, settings.abs_opt)
    @printf(io, "%stol_feas:      %.1e     tol_infeas:    %.1e\n", pad, settings.feas, settings.infeas)
    @printf(io, "%stol_illposed:  %.1e     tol_slow:      %.1e", pad, settings.illposed, settings.slow)
    return
end

function Base.show(io::IO, ::MIME"text/plain", settings::Settings{T}) where {T}
    println(io, "Settings{$T}:")
    show_settings(io, settings, 2)
end
