@kwdef mutable struct Settings{T}
    tol_rel_opt::T      = sqrt(eps(T))
    tol_abs_opt::T      = eps(T)^T(0.75)
    tol_feas::T         = sqrt(eps(T))
    tol_infeas::T       = eps(T)^T(0.75)
    tol_tau_infeas::T   = T(1e-2)
    tol_illposed::T     = eps(T)^T(0.75)
    tol_slow::T         = T(1e-3)
    near_factor::T      = T(1000)
    iter_limit::Int     = 1000
    prox_bound::T       = T(0.99)
    del_static::T  = zero(T)
    tol_dynamic::T = zero(T)
    del_dynamic::T = cbrt(eps(T))
    scaling::Bool       = true
    equilibration::Bool = true
    pivot::Bool         = false
    verbose::Bool       = true
end

function show_settings(io::IO, settings::Settings, indent::Int)
    pad = " "^indent
    @printf(io, "%sscaling:       %s\n", pad, settings.scaling ? "primal" : "dual")
    @printf(io, "%siter_limit:    %-7d     prox_bound:    %.2e\n", pad, settings.iter_limit, settings.prox_bound)
    @printf(io, "%stol_rel_opt:   %.1e     tol_abs_opt:   %.1e\n", pad, settings.tol_rel_opt, settings.tol_abs_opt)
    @printf(io, "%stol_feas:      %.1e     tol_infeas:    %.1e\n", pad, settings.tol_feas, settings.tol_infeas)
    @printf(io, "%stol_illposed:  %.1e     tol_slow:      %.1e", pad, settings.tol_illposed, settings.tol_slow)
    return
end

function Base.show(io::IO, ::MIME"text/plain", settings::Settings{T}) where {T}
    println(io, "Settings{$T}:")
    show_settings(io, settings, 2)
end
