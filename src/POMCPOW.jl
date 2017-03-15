module POMCPOW

using POMDPs
using POMCP
using ParticleFilters
using Parameters
using POMDPToolbox
using GenerativeModels

import POMCP: simulate, make_node
import Base: mean, rand, insert!
import POMDPs: action, solve

export
    POMCPOWSolver,
    POMCPPlanner2,
    POMCPOWTree,
    POWNodeBelief,
    CategoricalTree,
    CategoricalVector,

    MaxUCB,
    MaxQ,
    MaxTries,

    n_children,
    belief,

    blink

include("categorical_tree.jl")
include("categorical_vector.jl")
include("beliefs.jl")


include("tree.jl")
include("criteria.jl")

@with_kw immutable POMCPOWSolver <: AbstractPOMCPSolver
    eps::Float64                = 0.01
    max_depth::Int              = typemax(Int)
    criterion                   = MaxUCB(1.0)
    final_criterion             = MaxQ()
    tree_queries::Int           = 100
    rng::MersenneTwister        = Base.GLOBAL_RNG
    node_belief_updater         = POWNodeFilter()

    estimate_value::Any         = RolloutEstimator(RandomSolver(rng))

    enable_action_pw::Bool      = true
    check_repeat_obs::Bool      = true
    check_repeat_act::Bool      = true

    alpha_observation::Float64  = 0.5
    k_observation::Float64      = 10.0
    alpha_action::Float64       = 0.5
    k_action::Float64           = 10.0
    init_V::Float64             = 0.0
    init_N::Int                 = 0
    next_action::Any            = RandomActionGenerator(rng)
    default_action::Any         = ExceptionRethrow()
end

# unweighted ParticleCollections don't get anything pushed to them
function push_weighted!(::ParticleCollection, sp) end

include("solver.jl")

include("planner2.jl")
include("solver2.jl")

function solve(solver::POMCPOWSolver, problem::POMDP)
    return POMCPPlanner2(solver, problem)
end

include("visualization.jl")

end # module
