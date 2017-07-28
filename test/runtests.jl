using POMCPOW
using Base.Test

using POMDPModels
using POMDPToolbox

solver = POMCPOWSolver()

pomdp = BabyPOMDP()

test_solver(solver, pomdp, updater=updater(pomdp))

# make sure internal function is type stable
planner = solve(solver, pomdp)
b = initial_state_distribution(pomdp)
B = POMCPOW.belief_type(POMCPOW.POWNodeFilter, typeof(pomdp))
tree = POMCPOWTree{B,Bool,Bool,typeof(b)}(b, 2*planner.solver.tree_queries)
@inferred POMCPOW.simulate(planner, POMCPOW.POWTreeObsNode(tree, 1), true, 10)
