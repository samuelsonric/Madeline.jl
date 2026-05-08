struct Workspace{T, I}
    Mptr::FVector{I}
    Mval::FVector{T}
    Vval::FVector{T}
    Fval::FVector{T}
    Wval::FVector{T}
end

function Workspace{T}(S::ChordalSymbolic{I}, nrhs::Integer) where {T, I}
    nt = nthreads()

    if nt == 1
        Mptr = FVector{I}(undef, S.nMptr)
        Mval = FVector{T}(undef, max(S.nMval, S.nNval * nrhs))
        Vval = FVector{T}(undef, max(S.nMval, S.nNval * nrhs))
        Fval = FVector{T}(undef, max(S.nFval * S.nFval, S.nFval * nrhs))
    else
        bs = max(32, div(nrhs, 4nt))
        Mptr = FVector{I}(undef, S.nMptr * nt)
        Mval = FVector{T}(undef, max(S.nMval, S.nNval * bs * nt))
        Vval = FVector{T}(undef, max(S.nMval, S.nNval * bs * nt))
        Fval = FVector{T}(undef, max(S.nFval * S.nFval, S.nFval * bs * nt))
    end

    Wval = FVector{T}(undef, S.nFval * S.nFval)
    return Workspace(Mptr, Mval, Vval, Fval, Wval)
end
