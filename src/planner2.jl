mutable struct POMCPOWPlanner{P,NBU,C,NA,SE,IN,IV,SolverType} <: Policy
    solver::SolverType
    problem::P
    node_sr_belief_updater::NBU
    criterion::C
    next_action::NA
    solved_estimate::SE
    init_N::IN
    init_V::IV
    tree::Nullable{Any} # this is just so you can look at the tree later
end

function POMCPOWPlanner(solver, problem::POMDP)
    POMCPOWPlanner(solver,
                  problem,
                  solver.node_sr_belief_updater,
                  solver.criterion,
                  solver.next_action,
                  convert_estimator(solver.estimate_value, solver, problem),
                  solver.init_N,
                  solver.init_V,
                  Nullable{Any}())
end

Base.srand(p::POMCPOWPlanner, seed) = srand(p.solver.rng, seed)



function action_info{P,NBU}(pomcp::POMCPOWPlanner{P,NBU}, b)
    S = state_type(P)
    A = action_type(P)
    O = obs_type(P)
    B = belief_type(NBU,P)
    info = Dict{Symbol, Any}()
    tree = POMCPOWTree{B,A,O,typeof(b)}(b, 2*pomcp.solver.tree_queries)
    pomcp.tree = tree
    local a::A
    try
        a = search(pomcp, tree, info)
        if pomcp.solver.tree_in_info
            info[:tree] = tree
        end
    catch ex
        a = convert(A, default_action(pomcp.solver.default_action, pomcp.problem, b, ex))
    end
    return a, info
end

action(pomcp::POMCPOWPlanner, b) = first(action_info(pomcp, b))

function search(pomcp::POMCPOWPlanner, tree::POMCPOWTree, info::Dict{Symbol,Any}=Dict{Symbol,Any}())
    all_terminal = true
    # gc_enable(false)
    i = 0
    start_us = CPUtime_us()
    for i in 1:pomcp.solver.tree_queries
        s = rand(pomcp.solver.rng, tree.root_belief)
        if !POMDPs.isterminal(pomcp.problem, s)
            max_depth = min(pomcp.solver.max_depth, ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem))))
            simulate(pomcp, POWTreeObsNode(tree, 1), s, max_depth)
            all_terminal = false
        end
        if CPUtime_us() - start_us >= pomcp.solver.max_time*1e6
            break
        end
    end
    info[:search_time_us] = CPUtime_us() - start_us
    info[:tree_queries] = i

    if all_terminal
        throw(AllSamplesTerminal(tree.root_belief))
    end

    best_node = select_best(pomcp.solver.final_criterion, POWTreeObsNode(tree,1), pomcp.solver.rng)

    return tree.a_labels[best_node]
end
