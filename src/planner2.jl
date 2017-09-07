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

function action{P,NBU}(pomcp::POMCPOWPlanner{P,NBU}, b)
    S = state_type(P)
    A = action_type(P)
    O = obs_type(P)
    B = belief_type(NBU,P)
    tree = POMCPOWTree{B,A,O,typeof(b)}(b, 2*pomcp.solver.tree_queries)
    pomcp.tree = tree
    return search(pomcp, tree)
end

function search(pomcp::POMCPOWPlanner, tree::POMCPOWTree)
    all_terminal = true
    # gc_enable(false)
    start_time = CPUtime_us()
    for i in 1:pomcp.solver.tree_queries
        s = rand(pomcp.solver.rng, tree.root_belief)
        if !POMDPs.isterminal(pomcp.problem, s)
            simulate(pomcp, POWTreeObsNode(tree, 1), s, 0)
            all_terminal = false
        end
        if CPUtime_us() - start_time >= pomcp.solver.max_time*1e6
            break
        end
    end
    # gc_enable(true)

    if all_terminal
        throw(AllSamplesTerminal(tree.root_belief))
    end

    best_node = select_best(pomcp.solver.final_criterion, POWTreeObsNode(tree,1))

    return tree.a_labels[best_node]
end
