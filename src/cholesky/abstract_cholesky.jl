abstract type AbstractCholesky{T} end

const REGULARIZATION_EPSILON = 1e-12

function setzero! end
function addclique! end
function factorize! end
function ldiv_fwd! end
function ldiv_bwd! end
