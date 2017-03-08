using POMDPs
using POMCPOW
using ProfileView
using POMDPModels

#=
using Gallium
breakpoint(Pkg.dir("POMCPOW", "src", "solver2.jl"), 100)
=#

solver = POMCPOWSolver(tree_queries=50_000,
                     eps=0.01,
                     c=10.0,
                     enable_action_pw=false,
                     check_repeat_obs=false,
                     alpha_observation=1/8,
                     rng=MersenneTwister(2))

problem = LightDark1D()
policy = POMCPPlanner2(solver, problem)
ib = initial_state_distribution(problem)
a = action(policy, ib)

@time a = action(policy, ib)

Profile.clear()
@profile a = action(policy, ib)
ProfileView.view()

# @code_warntype POMCPOW.simulate(policy, 1, rand(Base.GLOBAL_RNG, ib), 10)
