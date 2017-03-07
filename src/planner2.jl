type POMCPPlanner2{S,A,O,SolverType} <: Policy
    solver::SolverType
    problem::POMDP{S,A,O}
    tree::Nullable{POMCPOWTree{POWNodeBelief{S,A,O},A,O}}
    solved_estimate::Any
end

POMCPPlanner2{S,A,O}(solver, problem::POMDP{S,A,O}) = POMCPPlanner2(solver, problem, Nullable{POMCPOWTree{POWNodeBelief{S,A,O},A,O}}(), convert_estimator(solver.estimate_value, solver, problem))

function action{S,A,O,Sol}(pomcp::POMCPPlanner2{S,A,O,Sol}, b)
    pomcp.tree = POMCPOWTree{POWNodeBelief{S,A,O},A,O}(b)
    return search(pomcp)
end

function search{S,A,O,Sol}(pomcp::POMCPPlanner2{S,A,O,Sol})
    tree = get(pomcp.tree)
    all_terminal = true
    for i in 1:pomcp.solver.tree_queries
        s = rand(pomcp.solver.rng, tree.root_belief)
        if !POMDPs.isterminal(pomcp.problem, s)
            simulate(pomcp, 1, s, 0)
            all_terminal = false
        end
    end

    if all_terminal
        throw(AllSamplesTerminal(tree.root_belief))
    end

    best_node = first(tree.tried[1])
    best_V = tree.v[best_node]
    @assert !isnan(best_V)
    for node in tree.tried[1][2:end]
        if tree.v[node] >= best_V
            best_V = tree.v[node]
            best_node = node
        end
    end

    return tree.a_labels[best_node]
end
