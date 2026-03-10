"""
    solve(prob::BasinVolumeProblem; explorer=nothing, n_burnin=500, kwargs...)

Estimate the volume of a basin of attraction.

Constructs a `VolumeEstimation.VolumeProblem` from the basin membership function and delegates
to `VolumeEstimation.jl` for the MCMC volume estimation.

By default, uses a `RandomWalkMH` explorer with step size tuned via a short burn-in
phase. This is much cheaper than the gradient-based `AutoMALA` default in VolumeEstimation.jl,
since each membership evaluation triggers an ODE solve.

# Keyword Arguments
- `explorer`: A Pigeons explorer to use. Default: `nothing` (creates a `RandomWalkMH`
  with burn-in-tuned step size).
- `n_burnin::Int`: Number of burn-in iterations for step size tuning. Default: `500`.
  Ignored if `explorer` is provided.
- `reference`: A Pigeons-compatible reference log potential. Overrides the default
  isotropic Gaussian reference. Must be used with `log_reference_volume`.
- `log_reference_volume::Union{Nothing,Float64}`: The known log partition function
  of the custom reference restricted to the membership region. Required when
  `reference` is provided.
- All remaining keyword arguments are forwarded to `VolumeEstimation.solve`, including:
  - `n_rounds::Int`: Number of PT adaptation rounds. Default: `10`.
  - `n_chains::Int`: Number of tempering chains. Default: `10`.

# Returns
A `VolumeEstimation.VolumeSolution` with fields `log_volume`, `volume`, and `pt`.
"""
function CommonSolve.solve(prob::BasinVolumeProblem; explorer=nothing, n_burnin::Int=500, kmax_options::NamedTuple=(;), reference=nothing, log_reference_volume::Union{Nothing,Float64}=nothing, kwargs...)
    # Cache membership to avoid double ODE solves at intermediate PT temperatures.
    # Pigeons evaluates both target(x) and reference(x) at the same point;
    # both call membership(x), so the cache saves one ODE solve per evaluation.
    membership = CachedMembership(prob.membership, prob.dim)
    if explorer === nothing
        baseline_sigma = burnin_step_size(membership, prob.x0; n_burnin=n_burnin)
        explorer = RandomWalkMH(initial_step_size=baseline_sigma)
    end
    vol_prob = VolumeProblem(membership, prob.dim; x0=prob.x0)
    result = CommonSolve.solve(vol_prob; explorer=explorer, kmax_options=kmax_options, reference=reference, log_reference_volume=log_reference_volume, kwargs...)
    _log_ode_stats(membership, result)
    return result
end

function _log_ode_stats(membership::CachedMembership, result)
    pt = result.pt

    # Aggregate across all thread-local slots (called single-threaded after solve)
    total_count = sum(s.total_count for s in membership.slots)
    total_count == 0 && return
    total_time = sum(s.total_time for s in membership.slots)

    solve_counts = Dict{Int, Int}()
    solve_times  = Dict{Int, Float64}()
    for slot in membership.slots
        for (chain, count) in slot.solve_counts
            solve_counts[chain] = get(solve_counts, chain, 0) + count
        end
        for (chain, t) in slot.solve_times
            solve_times[chain] = get(solve_times, chain, 0.0) + t
        end
    end

    betas = try
        pt.shared.tempering.schedule.grids
    catch
        nothing
    end

    lines = String[]

    # Per-chain exploration stats (only ODE solves inside step!)
    if !isempty(solve_counts)
        chains = sort(collect(keys(solve_counts)))
        if betas !== nothing
            sort!(chains, by=c -> get(betas, c, 0.0))
        end
        for chain in chains
            count = solve_counts[chain]
            t = solve_times[chain]
            avg_ms = count > 0 ? (t / count) * 1000 : 0.0
            label = if betas !== nothing && chain <= length(betas)
                "chain $chain (β=$(round(betas[chain], digits=4)))"
            else
                "chain $chain"
            end
            push!(lines, "  $label: $count solves, avg $(round(avg_ms, digits=3)) ms")
        end
        exploration_solves = sum(values(solve_counts))
        exploration_time   = sum(values(solve_times))
        push!(lines, "  exploration: $exploration_solves solves, $(round(exploration_time, digits=1)) s")
    end

    # Total across all phases (burn-in, kmax, exploration, swaps, sample_iid!)
    avg_ms = total_count > 0 ? (total_time / total_count) * 1000 : 0.0
    push!(lines, "  total: $total_count solves, avg $(round(avg_ms, digits=3)) ms, $(round(total_time, digits=1)) s wall")

    # Stepping stone estimator: log(Z_target / Z_ref) from Pigeons
    log_ratio = Pigeons.stepping_stone(pt)
    push!(lines, "  stepping stone log(Z_target/Z_ref): $(round(log_ratio, digits=4))")
    push!(lines, "  log(volume): $(round(result.log_volume, digits=4)),  volume: $(result.volume)")

    @info "ODE solve statistics\n" * join(lines, "\n")
end
