# this generated the error
#
# ERROR: LoadError: MethodError: no method matching init_node_sr_belief(::POMCPOW.POWNodeFilter, ::SimplePOMDP, ::Int64, ::Int64, ::Int64, ::Float64, ::Int64)

struct SimplePOMDP <: POMDP{Int,Int,Float64}
    p_success::Float64
end

function POMDPs.transition(m::SimplePOMDP, s::Int, a::Int)
    return SparseCat(clamp.([s + a, s - a], 1, 7), [m.p_success, 1 - m.p_success])
end

function POMDPs.observation(m::SimplePOMDP, a::Int, sp::Int)
    return SparseCat([sp + 0.5, sp - 0.5], [0.5, 0.5])
end

function POMDPs.reward(m::SimplePOMDP, s::Int, a::Int, sp::Int)
    if s == 1 || s == 7
        return 10
    elseif s == 4
        return -10
    else
        return 0
    end
end

function POMDPs.initialstate_distribution(m::SimplePOMDP)
    return SparseCat(1:7, ones(7) ./ 7)
end

function POMDPs.discount(m::SimplePOMDP)
    return 0.9
end

POMDPs.actions(m::SimplePOMDP) = [-1, 1]
POMDPs.states(m::SimplePOMDP) = 1:7
POMDPs.actionindex(m::SimplePOMDP, a::Int) = a == 1 ? 1 : 2

pomdp = SimplePOMDP(0.7)

s = 5
b = SparseCat([s], [1.0])
solver = POMCPOWSolver()

planner = solve(solver, pomdp)
a, info = action_info(planner, b)
