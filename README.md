# BasinVolumes.jl

> **Note:** This package is in it's beginning stages. Do not expect it to work in general cases yet

Estimate basin-of-attraction volumes for dynamical systems using parallel tempering.

## How it works

BasinVolumes.jl statistically estimates volumes of basins of attraction in high dimensions through the [VolumeEstimation.jl](https://github.com/spraharsh/VolumeEstimation.jl) package. It is done with the following approach

The approach:

1. Define a **membership function** `F(x0) -> Bool` that returns `true` if the trajectory starting at `x0` belongs to the target basin
2. The volume of the basin is the volume of the region `{x0 : F(x0) = true}`
3. Estimate this volume using parallel tempering MCMC (via [Pigeons.jl](https://github.com/Julia-Tempering/Pigeons.jl))

The MCMC scheme constructs a tempered path between a known reference distribution (a truncated Gaussian centered in the basin) and the uniform distribution over the basin. The stepping-stone estimator then recovers the log-volume ratio, giving the basin volume. This machinery lives in [VolumeEstimation.jl](https://github.com/spraharsh/VolumeEstimation.jl). Note that VolumeEstimation.jl has different performance considerations than BasinVolumes.jl because of the cost of solving an ODE every MC step. The defaults and functions in this library are tailored to that use case

## Installation

```julia
using Pkg
Pkg.add("BasinVolumes")
```

## Usage

### From an ODE system

The most common path: provide the ODE right-hand side, parameters, and a function that checks whether the final (steady) state is in the target basin.

```julia
using BasinVolumes

dim = 2
power = 0.5
offset = 1.0

# Gradient descent on a cosine potential with basins at integer lattice points
function neg_grad(u, p, t)
    S = dim + offset - sum(cos.(2pi .* u))
    -power .* S^(power - 1) .* (2pi) .* sin.(2pi .* u)
end

# Check convergence to the basin at the origin
basin_check(u) = sum(abs2, u) < 1e-6

prob = BasinVolumeProblem(neg_grad, nothing, basin_check, dim; x0=zeros(dim))
sol = solve(prob; n_rounds=8, n_chains=8)

sol.log_volume  # log of the estimated basin volume
sol.volume      # the estimated basin volume
```

The ODE constructor integrates until steady state (via `TerminateSteadyState` callback) and then applies the basin check. You can configure the solver, tolerances, and integration time:

```julia
prob = BasinVolumeProblem(f, p, basin_check, dim;
    x0 = zeros(dim),
    T = 1000.0,         # max integration time
    alg = Vern6(),       # ODE solver 
    ss_abstol = 1e-8,    # steady-state absolute tolerance
    ss_reltol = 1e-6,    # steady-state relative tolerance
    diffeq = (;),        # extra kwargs passed to ODE solve
)
```

### From a membership function directly

If you already have a boolean membership function (not necessarily from an ODE), you can use it directly:

```julia
# Volume of the unit ball in 2D (should be ~pi)
membership(x) = sum(abs2, x) <= 1.0
prob = BasinVolumeProblem(membership, 2)
sol = solve(prob; n_rounds=10, n_chains=10)
```

### Solver options

All keyword arguments beyond `explorer` and `n_burnin` are forwarded to `VolumeEstimation.solve`:

```julia
sol = solve(prob;
    n_rounds = 10,        # parallel tempering adaptation rounds
    n_chains = 10,        # number of tempering chains
    n_burnin = 500,       # burn-in iterations for step size tuning
    multithreaded = true, # run chains in parallel
)
```

By default, `solve` uses a `RandomWalkMH` explorer with step size tuned via a burn-in phase.

### Why RandomWalkMH

BasinVolumes.jl uses `RandomWalkMH` as its explorer for two reasons:

1. **Non-differentiable target.** The membership function is boolean — a point is either in the basin or not. There is no smooth log-density to differentiate, so gradient-based proposals like MALA/HMC cannot be used.
2. **Expensive evaluations.** Each membership check requires a full ODE integration to steady state. Random walk proposals only need one membership evaluation per step, while SliceSampler like methods would take multiple evaluations.

The step size is adaptively tuned per chain between PT rounds (target acceptance rate: 0.234, the asymptotic optimum for isotropic random walk Metropolis).

## Architecture

```
BasinVolumeProblem
    │
    ├─ ODE constructor: wraps f + basin_check into a membership function
    │  (integrates ODE to steady state, checks which basin it lands in)
    │
    └─ Direct constructor: takes a membership function directly
           │
           ▼
    CachedMembership (avoids redundant ODE solves)
           │
           ▼
    VolumeEstimation.VolumeProblem
           │
           ▼
    Pigeons.jl parallel tempering
           │
           ▼
    VolumeSolution { log_volume, volume, pt }
```

**Key components:**

- **`BasinVolumeProblem`** — problem definition, wrapping a membership function + dimension + starting point
- **`RandomWalkMH`** — Metropolis-Hastings explorer with per-chain adaptive step sizes (target acceptance: 0.234)
- **`CachedMembership`** — thread-safe caching layer that avoids double ODE solves when Pigeons evaluates both the target and reference distributions at the same point
- **`VolumeEstimation.jl`** — handles the actual volume estimation via parallel tempering and stepping-stone sampling
- **`Pigeons.jl`** — the parallel tempering engine

## Dependencies

- [CommonSolve.jl](https://github.com/SciML/CommonSolve.jl) — unified `solve()` interface
- [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) — ODE integration
- [Pigeons.jl](https://github.com/Julia-Tempering/Pigeons.jl) — parallel tempering MCMC
- [VolumeEstimation.jl](https://github.com/spraharsh/VolumeEstimation.jl) — volume estimation via tempered sampling
