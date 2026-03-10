module BasinVolumes

using CommonSolve
using DiffEqCallbacks
using OrdinaryDiffEqVerner
using Pigeons
using Statistics
using VolumeEstimation

_current_chain() = get(task_local_storage(), :_bv_current_chain, 0)
_set_current_chain!(c::Int) = task_local_storage(:_bv_current_chain, c)

include("problem.jl")
include("membership.jl")
include("explorer.jl")
include("cached_membership.jl")
include("solve.jl")

export BasinVolumeProblem, RandomWalkMH, solve

end # module BasinVolumes
