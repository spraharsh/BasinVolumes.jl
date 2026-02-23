"""
    BasinVolumeProblem(membership, dim; x0=zeros(dim))

Define a basin volume estimation problem.

The membership function `membership(x0) -> Bool` should return `true` if the
trajectory starting at initial condition `x0` converges to the target basin.

This wraps a `Volumes.VolumeProblem` under the hood — see `Volumes.jl` for details
on the MCMC volume estimation.

# Arguments
- `membership`: A function `f(x::AbstractVector) -> Bool` returning `true` if `x`
  is an initial condition in the target basin.
- `dim::Int`: Dimensionality of the state space.

# Keyword Arguments
- `x0::AbstractVector`: A point known to be in the basin, used for MCMC initialization.
  Default: `zeros(dim)`.
"""
struct BasinVolumeProblem{F}
    membership::F
    dim::Int
    x0::Vector{Float64}

    function BasinVolumeProblem(
        membership::F, dim::Int;
        x0::Union{Nothing, AbstractVector{<:Real}} = nothing
    ) where {F}
        x_init = x0 === nothing ? zeros(dim) : Vector{Float64}(x0)
        new{F}(membership, dim, x_init)
    end
end
