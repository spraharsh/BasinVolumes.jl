"""
    BasinVolumeProblem(f, p, basin_check, dim; x0=zeros(dim), T=1000.0, alg=Vern6(),
                       ss_abstol=1e-8, ss_reltol=1e-6, diffeq=(;))

Construct a `BasinVolumeProblem` from an ODE rule and a basin membership check.

This builds a membership function that:
1. Integrates the ODE `du/dt = f(u, p, t)` until steady state is reached
2. Checks whether the final state satisfies `basin_check(u_final) -> Bool`

Steady state is detected via a `TerminateSteadyState` callback which stops
integration when the derivatives are sufficiently small. The parameter `T`
acts as a maximum integration time safety limit.

# Arguments
- `f`: ODE right-hand side with signature `f(u, p, t) -> du` (out-of-place) or
  `f!(du, u, p, t)` (in-place).
- `p`: Parameters for the ODE.
- `basin_check`: A function `basin_check(u::AbstractVector) -> Bool` that returns `true`
  if the state `u` is in the target basin/attractor.
- `dim::Int`: Dimensionality of the state space.

# Keyword Arguments
- `x0::AbstractVector`: A point known to be in the basin. Default: `zeros(dim)`.
- `T::Real`: Maximum integration time (safety limit). Default: `1000.0`.
- `alg`: ODE solver algorithm. Default: `Vern6()`. Higher-order Vern methods
  work much better with `TerminateSteadyState` than lower-order methods like Tsit5.
- `ss_abstol::Real`: Absolute tolerance for steady-state detection. Default: `1e-8`.
- `ss_reltol::Real`: Relative tolerance for steady-state detection. Default: `1e-6`.
- `diffeq::NamedTuple`: Additional keyword arguments passed to `OrdinaryDiffEq.solve`.
  Default: `(;)`.
"""
function BasinVolumeProblem(
    f, p, basin_check, dim::Int;
    x0::Union{Nothing, AbstractVector{<:Real}} = nothing,
    T::Real = 1000.0,
    alg = Vern6(),
    ss_abstol::Real = 1e-8,
    ss_reltol::Real = 1e-6,
    diffeq = (;)
)
    x_init = x0 === nothing ? zeros(dim) : Vector{Float64}(x0)
    T_max = Float64(T)
    cb = TerminateSteadyState(ss_abstol, ss_reltol)

    membership = function (u0)
        ode_prob = ODEProblem(f, u0, (0.0, T_max), p)
        sol = CommonSolve.solve(ode_prob, alg; save_everystep=false, callback=cb, diffeq...)
        basin_check(sol.u[end])
    end

    BasinVolumeProblem(membership, dim; x0=x_init)
end
