"""
    CachedMembership(f, dim)

Wrap a membership function with a 1-entry cache to avoid redundant evaluations.

At intermediate parallel tempering temperatures, Pigeons evaluates both the target
and reference log potentials at the same point. Both call `membership(x)`, which
triggers an ODE solve. This wrapper caches the last result so the second call
at the same point returns immediately.

Also tracks ODE solve statistics (count and wall time) per PT chain via
the module-level `_CURRENT_CHAIN` ref, which is set by `RandomWalkMH` in `step!`.
"""
mutable struct CachedMembership{F}
    f::F
    last_x::Vector{Float64}
    last_result::Bool
    valid::Bool
    # ODE solve statistics per chain (exploration only, via _CURRENT_CHAIN)
    solve_counts::Dict{Int, Int}
    solve_times::Dict{Int, Float64}
    # Total ODE solve statistics (all phases: burn-in, kmax, exploration, swaps)
    total_count::Int
    total_time::Float64
end

CachedMembership(f, dim::Int) = CachedMembership(
    f, Vector{Float64}(undef, dim), false, false,
    Dict{Int, Int}(), Dict{Int, Float64}(),
    0, 0.0
)

function (cm::CachedMembership)(x)
    if cm.valid && length(x) == length(cm.last_x) && x == cm.last_x
        return cm.last_result
    end
    chain = _CURRENT_CHAIN[]
    t0 = time_ns()
    result = cm.f(x)
    elapsed_s = (time_ns() - t0) / 1e9
    cm.total_count += 1
    cm.total_time += elapsed_s
    if chain > 0
        cm.solve_counts[chain] = get(cm.solve_counts, chain, 0) + 1
        cm.solve_times[chain] = get(cm.solve_times, chain, 0.0) + elapsed_s
    end
    copyto!(resize!(cm.last_x, length(x)), x)
    cm.last_result = result
    cm.valid = true
    return result
end
