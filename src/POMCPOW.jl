module POMCPOW

using POMDPs
using BasicPOMCP
using ParticleFilters
using Parameters
using MCTS
using CPUTime
using D3Trees
using Colors
using Random
using Printf
using POMDPPolicies

using BasicPOMCP: convert_estimator

import Base: insert!
import POMDPs: action, solve, mean, rand, updater, currentobs, history
import POMDPModelTools: action_info

import MCTS: n_children, next_action, isroot, node_tag, tooltip_tag

export
    POMCPOWSolver,
    POMCPOWPlanner,
    POMCPOWTree,
    POWNodeBelief,
    POWTreeObsNode,
    CategoricalVector,
    FORollout,
    FOValue,

    MaxUCB,
    MaxQ,
    MaxTries,

    init_N,
    init_V,

    n_children,
    belief,
    sr_belief,
    isroot,

    POMCPOWVisualizer,
    blink

const init_V = init_Q

include("categorical_vector.jl")
include("beliefs.jl")

include("tree.jl")
include("criteria.jl")

"""
    POMCPOWSolver

Partially observable Monte Carlo planning with observation widening.

Fields:

- `eps::Float64`:
    Rollouts and tree expansion will stop when discount^depth is less than this.
    default: `0.01`
- `max_depth::Int`:
    Rollouts and tree expension will stop when this depth is reached.
    default: `10`
- `criterion::Any`:
    Criterion to decide which action to take at each node. e.g. `MaxUCB(c)`, `MaxQ`, or `MaxTries`
    default: `MaxUCB(1.0)`
- `final_criterion::Any`:
    Criterion for choosing the action to take after the tree is constructed.
    default: `MaxQ()`
- `tree_queries::Int`:
    Number of iterations during each action() call.
    default: `1000`
- `max_time::Float64`:
    Time limit for planning at each steps (seconds).
    default: `Inf`
- `rng::AbstractRNG`:
    Random number generator.
    default: Base.GLOBAL_RNG
- `node_sr_belief_updater::Updater`:
    Updater for state-reward distribution at the nodes.
    default: `POWNodeFilter()`
- `estimate_value::Any`: (rollout policy can be specified by setting this to RolloutEstimator(policy))
    Function, object, or number used to estimate the value at the leaf nodes.
    If this is a function `f`, `f(pomdp, s, h::BeliefNode, steps)` will be called to estimate the value.
    If this is an object `o`, `estimate_value(o, pomdp, s, h::BeliefNode, steps)` will be called.
    If this is a number, the value will be set to that number
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
- `tree_in_info::Bool`:
    If `true`, return the tree in the info dict when action_info is called, this can use a lot of memory if histories are being saved.
    default: `false`
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
    If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
    If this is a Policy `p`, `action(p, belief)` will be called.
    If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and
    if this method is not implemented, `a` will be returned directly.

"""
@with_kw mutable struct POMCPOWSolver{RNG<:AbstractRNG} <: AbstractPOMCPSolver
    eps::Float64                = 0.01
    max_depth::Int              = typemax(Int)
    criterion                   = MaxUCB(1.0)
    final_criterion             = MaxQ()
    tree_queries::Int           = 1000
    max_time::Float64           = Inf
    rng::RNG                    = Random.GLOBAL_RNG
    node_sr_belief_updater      = POWNodeFilter()

    estimate_value::Any         = RolloutEstimator(RandomSolver(rng))

    enable_action_pw::Bool      = true
    check_repeat_obs::Bool      = true
    check_repeat_act::Bool      = true
    tree_in_info::Bool          = false

    alpha_observation::Float64  = 0.5
    k_observation::Float64      = 10.0
    alpha_action::Float64       = 0.5
    k_action::Float64           = 10.0
    init_V::Any                 = 0.0
    init_N::Any                 = 0
    next_action::Any            = RandomActionGenerator(rng)
    default_action::Any         = ExceptionRethrow()
end

# unweighted ParticleCollections don't get anything pushed to them
function push_weighted!(::ParticleCollection, sp) end

include("planner2.jl")
include("solver2.jl")

function solve(solver::POMCPOWSolver, problem::POMDP)
    return POMCPOWPlanner(solver, problem)
end

include("updater.jl")
include("visualization.jl")

end # module
