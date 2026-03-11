"""
    RandomWalkMH(; initial_step_size=1.0, step_sizes=Dict{Int,Float64}(), n_steps=20, target_acceptance=0.234)

Random walk Metropolis-Hastings explorer for Pigeons.jl parallel tempering with basin volumes.
Since Basin volumes involve sampling over a flat distribution, Metropolis hasting is reasonable.
Adaptive samplers such as Slice Sampling that rely on multiple function evaluations per step
are unreasonable for volume calculations because tehy take too long

Step sizes are tuned **per-chain** via `adapt_explorer` between PT rounds, so each
temperature level gets its own scale.

# Fields
- `initial_step_size::Float64`: Baseline step size for chains not yet adapted. Default: `1.0`.
- `step_sizes::Dict{Int,Float64}`: Per-chain step sizes keyed by chain index.
- `n_steps::Int`: Number of MH proposals per exploration step. Default: `20` (keep low when
  each evaluation is expensive).
- `target_acceptance::Float64`: Target acceptance rate for adaptation. Default: `0.234`
  (asymptotically optimal for isotropic RWM).
"""
Base.@kwdef struct RandomWalkMH
    initial_step_size::Float64 = 1.0
    step_sizes::Dict{Int,Float64} = Dict{Int,Float64}()
    n_steps::Int = 20
    target_acceptance::Float64 = 0.234
end

function Pigeons.step!(explorer::RandomWalkMH, replica, shared)
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    state = replica.state
    rng = replica.rng
    chain = replica.chain
    _set_current_chain!(chain)
    dim = length(state)
    σ = get(explorer.step_sizes, chain, explorer.initial_step_size)

    current_lp = log_potential(state)

    for _ in 1:explorer.n_steps
        proposal = state .+ σ .* randn(rng, dim)
        proposed_lp = log_potential(proposal)

        log_alpha = proposed_lp - current_lp
        if log(rand(rng)) < log_alpha
            state .= proposal
            current_lp = proposed_lp
            Pigeons.@record_if_requested!(replica.recorders, :explorer_acceptance_pr, (chain, 1.0))
        else
            Pigeons.@record_if_requested!(replica.recorders, :explorer_acceptance_pr, (chain, 0.0))
        end
    end
    # Reset so that ODE solves outside exploration (swaps, AC tracking, sample_iid!)
    # are not misattributed to this chain.
    _set_current_chain!(0)
end

function Pigeons.adapt_explorer(explorer::RandomWalkMH, reduced_recorders, current_pt, new_tempering)
    new_step_sizes = copy(explorer.step_sizes)
    acc_stats = Pigeons.value(reduced_recorders.explorer_acceptance_pr)

    for (chain, stat) in pairs(acc_stats)
        acc_rate = Statistics.mean(stat)
        current_σ = get(new_step_sizes, chain, explorer.initial_step_size)
        ratio = clamp(acc_rate / explorer.target_acceptance, 0.5, 2.0)
        new_step_sizes[chain] = current_σ * ratio
    end

    RandomWalkMH(
        initial_step_size = explorer.initial_step_size,
        step_sizes = new_step_sizes,
        n_steps = explorer.n_steps,
        target_acceptance = explorer.target_acceptance,
    )
end

Pigeons.explorer_recorder_builders(::RandomWalkMH) = [Pigeons.explorer_acceptance_pr]

"""
    burnin_step_size(membership, x0; n_burnin=500, target_acceptance=0.234, adapt_interval=50)

Run a short adaptive burn-in on the target distribution to find a good baseline
step size for `RandomWalkMH`.

Since the target is uniform on the membership region, MH acceptance simplifies to:
accept if `membership(proposal)` is true.

Returns the tuned step size `σ`.
"""
function burnin_step_size(membership, x0;
        n_burnin::Int = 500,
        target_acceptance::Float64 = 0.234,
        adapt_interval::Int = 50)
    dim = length(x0)
    σ = 1.0
    x = copy(x0)
    n_accept = 0

    for i in 1:n_burnin
        proposal = x .+ σ .* randn(dim)
        if membership(proposal)
            x .= proposal
            n_accept += 1
        end
        if i % adapt_interval == 0
            acc_rate = n_accept / adapt_interval
            σ *= exp(acc_rate - target_acceptance)
            n_accept = 0
        end
    end
    return σ
end
