#=
Kuramoto Basin Volume Verification
====================================
Verifies the Gaussian basin-size scaling e^{-kq²} for n identical Kuramoto
oscillators on a cycle graph (Groisman et al., PRE 2025).

The stable attractors for n oscillators are twisted states with winding numbers
|q| < n/4 (strictly).  We compute log V(q) for q = 0 … q_max and check that
  log V(q) - log V(0) ≈ -kq²
i.e., (log V(q) - log V(0)) / q² is constant in q.

q=0 (the sync basin) is included as the reference: its volume on T^n is
well-defined and should be the largest basin.

Usage:
    julia --project=. kuramoto/basin_volumes_kuramoto.jl
    julia -p 4 --project=. kuramoto/basin_volumes_kuramoto.jl  # explicit workers
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Distributed

const N = 10   # number of oscillators; stable twisted states: 0 ≤ |q| < N/4 = 2.5
const q_range = 0:(ceil(Int, N / 4) - 1)

# Add one worker per q value if not already running with -p
if nworkers() == 1
    addprocs(length(q_range))
end

println("Running on $(nworkers()) workers for q ∈ {$(join(q_range, ", "))}")

# ── Load packages and define all functions on every worker ────────────────────

@everywhere begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using BasinVolumes
    using CommonSolve
    using DifferentialEquations

    # Kuramoto cycle: θ̇_j = sin(θ_{j+1}-θ_j) + sin(θ_{j-1}-θ_j), periodic BC
    function kuramoto!(dθ, θ, _p, _t)
        n = length(θ)
        for j in 1:n
            dθ[j] = sin(θ[mod1(j+1, n)] - θ[j]) + sin(θ[mod1(j-1, n)] - θ[j])
        end
    end

    twisted_state(q, n) = [2π * q * j / n for j in 1:n]

    function is_in_basin(θ_final, q, n; tol=1e-2)
        θ_rot    = θ_final .- θ_final[1]
        θ_target = [2π * q * (j - 1) / n for j in 1:n]
        d = sqrt(sum((mod(θ_rot[j] - θ_target[j] + π, 2π) - π)^2 for j in 1:n))
        return d < tol
    end

    function make_membership(q_target, n)
        x0 = twisted_state(q_target, n)

        function membership(θ_raw)
            if any(abs(θ_raw[j] - x0[j]) > π for j in eachindex(θ_raw))
                return false
            end
            θ = mod.(θ_raw, 2π)
            prob = ODEProblem(kuramoto!, θ, (0.0, 10000.0), nothing)
            cb   = TerminateSteadyState(1e-8, 1e-6)
            sol  = CommonSolve.solve(prob, Vern6(); save_everystep=false, callback=cb)
            return is_in_basin(sol.u[end], q_target, n)
        end

        return membership, x0
    end

    function compute_basin_volume(q, n; n_rounds=12, n_chains=30)
        println("  [worker $(myid())] starting q=$q")
        membership, x0 = make_membership(q, n)
        prob = BasinVolumeProblem(membership, n; x0=x0)
        sol  = solve(prob; n_rounds=n_rounds, n_chains=n_chains, multithreaded=true)
        lv = sol.log_volume
        println("  [worker $(myid())] done    q=$q  log_volume=$lv")
        return lv
    end
end

# ── Sanity-check all attractors before running (on main process) ──────────────

println("N=$N oscillators  →  stable q ∈ {$(join(q_range, ", "))}")
println("Verifying twisted states are in their own basins...")
for q in q_range
    membership, x0 = make_membership(q, N)
    @assert membership(x0) "Twisted state q=$q is not identified as in its own basin!"
    println("  q=$q ✓")
end
println()

# ── Basin volume computation — each q on a separate worker ───────────────────

println("Dispatching q=$(q_range) across $(nworkers()) workers...")
futures = [Distributed.@spawnat :any compute_basin_volume(q, N) for q in q_range]

# Block until all finish and collect results
results = Dict(q => fetch(f) for (q, f) in zip(q_range, futures))
println()

# ── Gaussian scaling check ────────────────────────────────────────────────────
# If log V(q) = C - k q², then (log V(q) - log V(0)) / q² = -k (constant).

println("=" ^ 55)
println("Gaussian scaling verification  (e^{-kq²} conjecture)")
println("=" ^ 55)
println("  Expect: [log V(q) - log V(0)] / q² = -k  (constant)")
println()

lv0 = results[0]
println("  q=0  log_volume=$(round(lv0; digits=3))  (reference, sync basin)")
ks = Float64[]
for q in 1:maximum(q_range)
    !haskey(results, q) && continue
    Δ = results[q] - lv0
    k_est = -Δ / q^2
    push!(ks, k_est)
    println("  q=$q  log_volume=$(lpad(round(results[q]; digits=3), 9))  Δ=$(lpad(round(Δ; digits=3), 9))  k_est=$(round(k_est; digits=4))")
end

if !isempty(ks)
    println()
    k_mean = sum(ks) / length(ks)
    k_std  = length(ks) > 1 ? sqrt(sum((k - k_mean)^2 for k in ks) / (length(ks)-1)) : 0.0
    println("  k = $(round(k_mean; digits=4)) ± $(round(k_std; digits=4))")
    if k_std < 0.3 * abs(k_mean)
        println("✓  Gaussian scaling confirmed (k is consistent across q)")
    else
        println("✗  k varies with q — may need more rounds or larger n")
    end
end
