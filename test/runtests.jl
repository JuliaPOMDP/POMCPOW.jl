using POMCPOW
using Base.Test

using POMDPModels
using POMDPToolbox

solver = POMCPOWSolver()

pomdp = BabyPOMDP()

test_solver(solver, pomdp, updater=updater(pomdp))
