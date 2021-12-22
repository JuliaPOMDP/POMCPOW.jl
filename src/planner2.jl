mutable struct POMCPOWPlanner{P,NBU,C,NA,SE,IN,IV,SolverType} <: Policy
    solver::SolverType
    problem::P
    node_sr_belief_updater::NBU
    criterion::C
    next_action::NA
    solved_estimate::SE
    init_N::IN
    init_V::IV
    tree::Union{Nothing, POMCPOWTree} # this is just so you can look at the tree later
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
                  nothing)
end

Random.seed!(p::POMCPOWPlanner, seed) = Random.seed!(p.solver.rng, seed)

function action_info(pomcp::POMCPOWPlanner{P,NBU}, b; tree_in_info=false) where {P,NBU}
    A = actiontype(P)
    info = Dict{Symbol, Any}()
    tree = make_tree(pomcp, b)
    pomcp.tree = tree
    local a::A
    try
        a = search(pomcp, tree, info)
        if pomcp.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
    catch ex
        a = convert(A, default_action(pomcp.solver.default_action, pomcp.problem, b, ex))
    end
    return a, info
end

action(pomcp::POMCPOWPlanner, b) = first(action_info(pomcp, b))

function POMDPPolicies.actionvalues(p::POMCPOWPlanner, b)
    tree = make_tree(p, b)
    search(p, tree)
    values = Vector{Union{Float64,Missing}}(missing, length(actions(p.problem)))
    for anode in tree.tried[1]
        a = tree.a_labels[anode]
        values[actionindex(p.problem, a)] = tree.v[anode]
    end
    return values
end

function make_tree(p::POMCPOWPlanner{P, NBU}, b) where {P, NBU}
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    B = belief_type(NBU,P)
    return POMCPOWTree{B, A, O, typeof(b)}(b, 2*min(100_000, p.solver.tree_queries))
    # return POMCPOWTree{B, A, O, typeof(b)}(b, 2*p.solver.tree_queries)
end


function search(pomcp::POMCPOWPlanner, tree::POMCPOWTree, info::Dict{Symbol,Any}=Dict{Symbol,Any}())
    timer = pomcp.solver.timer
    all_terminal = true
    # gc_enable(false)
    i = 0
    t0 = timer()
    while i < pomcp.solver.tree_queries
        i += 1
        s = rand(pomcp.solver.rng, tree.root_belief)
        if !POMDPs.isterminal(pomcp.problem, s)
            max_depth = min(pomcp.solver.max_depth, ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem))))
            simulate(pomcp, POWTreeObsNode(tree, 1), s, max_depth)
            all_terminal = false
        end
        if timer() - t0 >= pomcp.solver.max_time
            break
        end
    end
    info[:search_time] = timer() - t0
    info[:tree_queries] = i

    if all_terminal
        throw(AllSamplesTerminal(tree.root_belief))
    end

    best_node = select_best(pomcp.solver.final_criterion, POWTreeObsNode(tree,1), pomcp.solver.rng)

    return tree.a_labels[best_node]
end
