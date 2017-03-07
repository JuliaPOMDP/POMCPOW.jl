function make_node{S,A,O,B}(pomcp::POMCPPlanner{S,A,O,B,POMCPOWSolver}, belief)
    ANodeType = POWActNode{A, O, POWObsNode{B,A,O}}
    return RootNode{typeof(belief), A, ANodeType}(0, belief, Dict{A,ANodeType}())
end

function simulate{S,A,O,B}(pomcp::POMCPPlanner{S,A,O,B,POMCPOWSolver}, h::BeliefNode, s::S, depth)

    sol = pomcp.solver

    if POMDPs.discount(pomcp.problem)^depth < sol.eps ||
            POMDPs.isterminal(pomcp.problem, s) ||
            depth >= sol.max_depth
        return 0
    end

    total_N = reduce(add_N, 0, values(h.children))
    if sol.enable_action_pw
        if length(h.children) <= sol.k_action*total_N^sol.alpha_action
            a = next_action(sol.next_action, pomcp.problem, h.B, h) # XXX should this be a function of s or h.B?
            if !(a in keys(h.children))
                h.children[a] = POWActNode(a,
                                    init_N(sol.init_N, pomcp.problem, h, a),
                                    init_V(sol.init_V, pomcp.problem, h, a),
                                    0,
                                    Vector{O}(),
                                    Dict{O,POWObsNode{B,A,O}}())
            end
            if length(h.children) <= 1
                if depth > 0
                    return POMDPs.discount(pomcp.problem)^depth * estimate_value(pomcp.solved_estimate, pomcp.problem, s, h, depth)
                else
                    return 0.
                end
            end
        end
    else # run through all the actions
        if isempty(h.children)
            action_space_iter = POMDPs.iterator(POMDPs.actions(pomcp.problem))
            for a in action_space_iter
                h.children[a] = POWActNode(a,
                                init_N(sol.init_N, pomcp.problem, h, a),
                                init_V(sol.init_V, pomcp.problem, h, a),
                                0,
                                Vector{O}(),
                                Dict{O,POWObsNode{B,A,O}}())
            end

            if depth > 0 # no need for a rollout if this is the root node
                return POMDPs.discount(pomcp.problem)^depth * estimate_value(pomcp.solved_estimate, pomcp.problem, s, h, depth)
            else
                return 0.
            end
        end
        total_N = h.N
    end

    # Calculate UCT
    best_criterion_val = -Inf
    local best_node
    for (a,node) in h.children
        if node.N == 0 && total_N <= 1
            criterion_value = node.V
        elseif node.N == 0 && node.V == -Inf
            criterion_value = Inf
        else
            criterion_value = node.V + sol.c*sqrt(log(total_N)/node.N)
        end
        if criterion_value >= best_criterion_val
            best_criterion_val = criterion_value
            best_node = node
        end
    end
    a = best_node.label

    sp, o, r = GenerativeModels.generate_sor(pomcp.problem, s, a, sol.rng)

    # if length(best_node.children) <= sol.k_observation*(best_node.N^sol.alpha_observation)
    if best_node.n_children <= sol.k_observation*(best_node.N^sol.alpha_observation)

        push!(best_node.generated, o)

        if haskey(best_node.children, o)
            hao = best_node.children[o]
        else
            if isa(pomcp.node_belief_updater, ParticleReinvigorator)
                hao = POWObsNode(o, 0, ParticleCollection{S}(), Dict{A,POWActNode{A,O,POWObsNode{B,A,O}}}())
                push!(hao.B, sp)
            elseif isa(pomcp.node_belief_updater, POWNodeFilter)
                hao = POWObsNode(o, 0, POWNodeBelief(pomcp.problem, s, a, o),
                              Dict{A,POWActNode{A,O,POWObsNode{B,A,O}}}())
                push_weighted!(hao.B, sp)
            else
                new_belief = update(pomcp.node_belief_updater, h.B, a, o) # this relies on h.B not being modified
                hao = POWObsNode(o, 0, new_belief, Dict{A,POWActNode{A,O,POWObsNode{B,A,O}}}())
            end
            best_node.children[o] = hao
            best_node.n_children += 1
        end

        hao.N += 1
    else
        o = rand(sol.rng, best_node.generated)
        hao = best_node.children[o]
        push_weighted!(hao.B, sp)
        sp = rand(sol.rng, hao.B)
        r = POMDPs.reward(pomcp.problem, s, a, sp) # should cache this so the user doesn't have to implement reward
    end

    if r == Inf
        warn("POMCP: +Inf reward. This is not recommended and may cause future errors.")
    end

    R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, hao, sp, depth+1)

    best_node.N += 1
    if best_node.V != -Inf
        best_node.V += (R-best_node.V)/best_node.N
    end

    return R
end

