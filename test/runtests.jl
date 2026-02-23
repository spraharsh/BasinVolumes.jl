using Test
using BasinVolumes
using CommonSolve

@testset "BasinVolumes" begin
    @testset "Direct membership" begin
        # Unit ball in 2D: area = pi
        membership(x) = sum(abs2, x) <= 1.0
        prob = BasinVolumeProblem(membership, 2)
        sol = solve(prob; n_rounds=10, n_chains=10)
        @test sol.volume > 0
        @test isapprox(sol.volume, pi; rtol=0.5)
    end

    @testset "PoweredCosineSum basin volume" begin
        # PoweredCosineSum potential: V(x) = (d + offset - sum cos(2π xᵢ))^power
        # Gradient descent ODE: du/dt = -∇V(u)
        # Each basin is a unit hypercube centered at an integer lattice point.
        # Basin volume = 1, so log(volume) = 0.
        dim = 2
        power = 0.5
        offset = 1.0

        function neg_grad_powsumcos(u, p, t)
            S = dim + offset - sum(cos.(2π .* u))
            -power .* S^(power - 1) .* (2π) .* sin.(2π .* u)
        end

        # Check if trajectory converged near the origin (the steady state)
        basin_check(u) = sum(abs2, u) < 1e-6

        prob = BasinVolumeProblem(neg_grad_powsumcos, nothing, basin_check, dim;
            x0=zeros(dim))
        sol = solve(prob; n_rounds=8, n_chains=8, n_burnin=100, kmax_options=(n_samples=100,), multithreaded=true)
        # Basin volume = 1, so log(volume) should be 0
        @test isapprox(sol.log_volume, 0.0; atol=0.05)
    end
end
