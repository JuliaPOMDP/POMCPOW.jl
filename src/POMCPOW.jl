module POMCPOW

using POMDPs
using POMCP
using ParticleFilters
using Parameters
using POMDPToolbox
using GenerativeModels

import POMCP: simulate, make_node
import Base: mean, rand

export
    POMCPOWSolver

type POWNodeBelief{S,A,O}
    model::POMDP{S,A,O}
    s::S
    a::A
    o::O
    particles::Vector{S}
    weights::Vector{Float64}
    weight_sum::Float64
end
POWNodeBelief{S,A,O}(model, s::S, a::A, o::O) = POWNodeBelief{S,A,O}(model, s, a, o, S[], Float64[], 0.0)

immutable POWNodeFilter <: Updater{POWNodeBelief} end

@with_kw type POMCPOWSolver <: AbstractPOMCPSolver
    eps::Float64                = 0.01
    max_depth::Int              = typemax(Int)
    c::Float64                  = 1
    tree_queries::Int           = 100
    rng::AbstractRNG            = Base.GLOBAL_RNG
    node_belief_updater::Union{Updater, DefaultReinvigoratorStub} = POWNodeFilter()

    estimate_value::Any         = RolloutEstimator(RandomSolver())

    enable_action_pw::Bool      = true

    alpha_observation::Float64  = 0.5
    k_observation::Float64      = 10.0
    alpha_action::Float64       = 0.5
    k_action::Float64           = 10.0
    init_V::Any                 = 0.0
    init_N::Any                 = 0
    next_action::Any            = RandomActionGenerator()
    default_action::Any         = ExceptionRethrow()
end

function push_weighted!(b::POWNodeBelief, sp)
    w = obs_weight(b.model, b.s, b.a, sp, b.o)
    push!(b.particles, sp)
    push!(b.weights, w)
    b.weight_sum += w
end

rand(rng::AbstractRNG, b::POWNodeBelief) = rand(rng, WeightedParticleBelief(b.particles, b.weights, b.weight_sum))
mean(b::POWNodeBelief) = sum(b.particles.*b.weights)/sum(b.weights)

# unweighted ParticleCollections don't get anything pushed to them
function push_weighted!(::ParticleCollection, sp) end

type POMCPPlanner2{SolverType} <: Policy
    problem::POMDP
    solver::SolverType
end

include("tree.jl")
include("solver.jl")

end # module
