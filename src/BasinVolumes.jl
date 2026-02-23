module BasinVolumes

using CommonSolve
using DifferentialEquations
using Pigeons
using Statistics
using Volumes

# Track which PT chain is currently evaluating (for ODE stats)
const _CURRENT_CHAIN = Ref(0)

include("problem.jl")
include("membership.jl")
include("explorer.jl")
include("cached_membership.jl")
include("solve.jl")

export BasinVolumeProblem, RandomWalkMH, solve

end # module BasinVolumes
