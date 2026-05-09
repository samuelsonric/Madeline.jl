struct Workspace{T, I}
    Mptr::FVector{I}
    Mval::FVector{T}
    Vval::FVector{T}
    Fval::FVector{T}
    Wval::FVector{T}
end

function Workspace{T}(S::ChordalSymbolic{I}, nrhs::Integer) where {T, I}
    Mptr = FVector{I}(undef, S.nMptr)
    Mval = FVector{T}(undef, max(S.nMval, S.nNval * nrhs))
    Vval = FVector{T}(undef, max(S.nMval, S.nNval * nrhs))
    Fval = FVector{T}(undef, max(S.nFval * S.nFval, S.nFval * nrhs))
    Wval = FVector{T}(undef, S.nFval * S.nFval)
    return Workspace(Mptr, Mval, Vval, Fval, Wval)
end
