module POMCPOW

using POMDPs
using POMCP
using ParticleFilters
using Parameters
using POMDPToolbox
using GenerativeModels

import POMCP: simulate, make_node
import Base: mean, rand, insert!
import POMDPs: action

export
    POMCPOWSolver,
    POMCPPlanner2,
    CategoricalTree,
    CategoricalVector

include("categorical_tree.jl")
include("categorical_vector.jl")

immutable POWNodeBelief{S,A,O,P}
    model::P
    s::S
    a::A
    o::O
    dist::CategoricalTree{S}

    POWNodeBelief(m,s,a,o,d) = new(m,s,a,o,d)
    function POWNodeBelief(m::P, s::S, a::A, o::O, sp::S)
        new(m, s, a, o, CategoricalTree{S}(sp, obs_weight(m, s, a, sp, o)))
    end
end

function POWNodeBelief{S,A,O}(model::POMDP{S,A,O}, s::S, a::A, o::O, sp::S)
    POWNodeBelief{S,A,O,typeof(model)}(model, s, a, o,
                         CategoricalTree{S}(sp, obs_weight(model, s, a, sp, o)))
end

immutable POWNodeFilter <: Updater{POWNodeBelief} end

@with_kw type POMCPOWSolver <: AbstractPOMCPSolver
    eps::Float64                = 0.01
    max_depth::Int              = typemax(Int)
    c::Float64                  = 1
    tree_queries::Int           = 100
    rng::MersenneTwister        = Base.GLOBAL_RNG
    node_belief_updater::Union{Updater, DefaultReinvigoratorStub} = POWNodeFilter()

    estimate_value::Any         = RolloutEstimator(RandomSolver())

    enable_action_pw::Bool      = true
    check_repeat_obs::Bool      = true
    check_repeat_act::Bool      = true

    alpha_observation::Float64  = 0.5
    k_observation::Float64      = 10.0
    alpha_action::Float64       = 0.5
    k_action::Float64           = 10.0
    init_V::Float64             = 0.0
    init_N::Int                 = 0
    next_action::Any            = RandomActionGenerator()
    default_action::Any         = ExceptionRethrow()
end

function push_weighted!(b::POWNodeBelief, sp)
    w = obs_weight(b.model, b.s, b.a, sp, b.o)
    insert!(b.dist, sp, w)
end

rand(rng::AbstractRNG, b::POWNodeBelief) = rand(rng, b.dist)
mean(b::POWNodeBelief) = sum(b.particles.*b.weights)/sum(b.weights)

# unweighted ParticleCollections don't get anything pushed to them
function push_weighted!(::ParticleCollection, sp) end


include("tree.jl")
include("solver.jl")

include("planner2.jl")
include("solver2.jl")

end # module
