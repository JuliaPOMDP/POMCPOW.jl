using CommonRLSpaces
using Distributions
using LinearAlgebra
using StaticArrays

struct D513POMDP <: POMDP{SVector{3,Float64}, SVector{2,Float64}, SVector{3,Float64}} end

POMDPs.states(m::D513POMDP) = Box([-5,-5,-3], [5,5,3])
POMDPs.actions(m::D513POMDP) = Box([-5,-5], [5,5])
POMDPs.observations(m::D513POMDP) = Box([-5,-5,-3], [5,5,3])
POMDPs.transition(m::D513POMDP, s, a, dt=0.1) = MvNormal(s, Diagonal([0.1,0.1,0.1]))
POMDPs.observation(m::D513POMDP, s, a, sp) = MvNormal(sp, Diagonal([0.001,0.001,0.001]))
POMDPs.reward(m::D513POMDP, s, a, sp) = 0
POMDPs.discount(m::D513POMDP) = 0.9

m = D513POMDP()
solver = POMCPOWSolver()
policy = solve(solver, m)
a = action(policy, Deterministic([0.0,0.0,0.0]))
@test a in actions(m)
