# POMCPOW

[![Build Status](https://travis-ci.org/JuliaPOMDP/POMCPOW.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/POMCPOW.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaPOMDP/POMCPOW.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaPOMDP/POMCPOW.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaPOMDP/POMCPOW.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaPOMDP/POMCPOW.jl?branch=master)

POMCPOW is an online solver based on Monte Carlo tree search for POMDPs with continuous state, action, and observation spaces. For more information, see https://arxiv.org/abs/1709.06196.

It solves problems specified using the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface. The requirements are the same as for an importance-sampling particle filter - a generative model for the dynamics and an explicit observation model.

# Installation

For Julia 1.0 and above, use the JuliaPOMDP registry:

```julia
import Pkg
Pkg.add("POMDPs")
import POMDPs
POMDPs.add_registry()
Pkg.add("POMCPOW")
```

# Usage

```julia
using POMDPs
using POMCPOW
using POMDPModels
using POMDPSimulators
using POMDPPolicies

solver = POMCPOWSolver(criterion=MaxUCB(20.0))
pomdp = BabyPOMDP() # from POMDPModels
planner = solve(solver, pomdp)

hr = HistoryRecorder(max_steps=100)
hist = simulate(hr, pomdp, planner)
for (s, b, a, r, sp, o) in hist
    @show s, a, r, sp
end

rhist = simulate(hr, pomdp, RandomPolicy(pomdp))
println("""
    Cumulative Discounted Reward (for 1 simulation)
        Random: $(discounted_reward(rhist))
        POMCPOW: $(discounted_reward(hist))
    """)
```

Algorithm options are controlled with keyword arguments to the constructor. Use `?POMCPOWSolver` to see a list of options. It should output the following:

Fields:

- `eps::Float64`:
    Rollouts and tree expansion will stop when discount^depth is less than this.
    default: `0.01`
- `max_depth::Int`:
    Rollouts and tree expension will stop when this depth is reached.
    default: `10`
- `criterion::Any`:
    Criterion to decide which action to take at each node. e.g. `MaxUCB(c)`, `MaxQ`, or `MaxTries`.
    default: `MaxUCB(1.0)`
- `final_criterion::Any`:
    Criterion for choosing the action to take after the tree is constructed.
    default: `MaxQ()`
- `tree_queries::Int`:
    Number of iterations during each action() call.
    default: `100`
- `max_time::Float64`:
    Time limit for planning at each steps (seconds).
    default: `Inf`
- `rng::AbstractRNG`:
    Random number generator.
    default: `Base.GLOBAL_RNG`
- `node_sr_belief_updater::Updater`:
    Updater for state-reward distribution at the nodes.
    default: `POWNodeFilter()`
- `estimate_value::Any`: (rollout policy can be specified by setting this to RolloutEstimator(policy))
    Function, object, or number used to estimate the value at the leaf nodes.
    If this is a function `f`, `f(pomdp, s, h::BeliefNode, steps)` will be called to estimate the value.
    If this is an object `o`, `estimate_value(o, pomdp, s, h::BeliefNode, steps)` will be called.
    If this is a number, the value will be set to that number.
    default: `RolloutEstimator(RandomSolver(rng))`
- `enable_action_pw::Bool`:
    Controls whether progressive widening is done on actions; if `false`, the entire action space is used.
    default: `true`
- `check_repeat_obs::Bool`:
    Check if an observation was sampled multiple times. This has some dictionary maintenance overhead, but prevents multiple nodes with the same observation from being created. If the observation space is discrete, this should probably be used, but can be turned off for speed.
    default: `true`
- `check_repeat_act::Bool`:
    Check if an action was sampled multiple times. This has some dictionary maintenance overhead, but prevents multiple nodes with the same action from being created. If the action space is discrete, this should probably be used, but can be turned off for speed.
    default: `true`
- `k_action::Float64`, `alpha_action::Float64`, `k_observation::Float64`, `alpha_observation::Float64`:
    These constants control the double progressive widening. A new observation
    or action will be added if the number of children is less than or equal to kN^alpha.
    defaults: k: `10`, alpha: `0.5`
- `init_V::Any`:
    Function, object, or number used to set the initial V(h,a) value at a new node.
    If this is a function `f`, `f(pomdp, h, a)` will be called to set the value.
    If this is an object `o`, `init_V(o, pomdp, h, a)` will be called.
    If this is a number, V will be set to that number
    default: `0.0`
- `init_N::Any`:
    Function, object, or number used to set the initial N(s,a) value at a new node.
    If this is a function `f`, `f(pomdp, h, a)` will be called to set the value.
    If this is an object `o`, `init_N(o, pomdp, h, a)` will be called.
    If this is a number, N will be set to that number
    default: `0`
- `next_action::Any`
    Function or object used to choose the next action to be considered for progressive widening.
    The next action is determined based on the POMDP, the belief, `b`, and the current `BeliefNode`, `h`.
    If this is a function `f`, `f(pomdp, b, h)` will be called to set the value.
    If this is an object `o`, `next_action(o, pomdp, b, h)` will be called.
    default: `RandomActionGenerator(rng)`
- `default_action::Any`:
    Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
    If this is a Function `f`, `f(belief, ex)` will be called.
    If this is a Policy `p`, `action(p, belief)` will be called.
    If it is an object `a`, `default_action(a, belief, ex)` will be called, and
    if this method is not implemented, `a` will be returned directly.


Check out [VDPTag2.jl](https://github.com/zsunberg/VDPTag2.jl/blob/master/README.md) for an additional problem that is solved by POMCPOW. 
