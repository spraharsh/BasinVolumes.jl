"""
    CachedMembership(f, dim)

Wrap a membership function with per-thread independent cache+stats slots.

Each thread owns exactly one `_CacheSlot` and never touches another thread's slot,
so there is no shared mutable state and no locking. Stats are aggregated across
slots only after `solve` returns (single-threaded context).

The cache avoids redundant ODE solves when Pigeons evaluates both the target and
reference log potentials at the same point within a single chain step.
"""
mutable struct _CacheSlot
    last_x::Vector{Float64}
    last_result::Bool
    valid::Bool
    solve_counts::Dict{Int, Int}
    solve_times::Dict{Int, Float64}
    total_count::Int
    total_time::Float64
end

_CacheSlot(dim::Int) = _CacheSlot(
    Vector{Float64}(undef, dim), false, false,
    Dict{Int, Int}(), Dict{Int, Float64}(),
    0, 0.0
)

struct CachedMembership{F}
    f::F
    slots::Vector{_CacheSlot}
end

CachedMembership(f, dim::Int) = CachedMembership(
    f, [_CacheSlot(dim) for _ in 1:Threads.nthreads()]
)

function (cm::CachedMembership)(x)
    slot = cm.slots[Threads.threadid()]
    if slot.valid && length(x) == length(slot.last_x) && x == slot.last_x
        return slot.last_result
    end
    chain = _current_chain()
    t0 = time_ns()
    result = cm.f(x)
    elapsed_s = (time_ns() - t0) / 1e9
    slot.total_count += 1
    slot.total_time += elapsed_s
    if chain > 0
        slot.solve_counts[chain] = get(slot.solve_counts, chain, 0) + 1
        slot.solve_times[chain] = get(slot.solve_times, chain, 0.0) + elapsed_s
    end
    copyto!(resize!(slot.last_x, length(x)), x)
    slot.last_result = result
    slot.valid = true
    return result
end
