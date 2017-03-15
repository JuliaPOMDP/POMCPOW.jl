function simulate{B,S,A,O}(pomcp::POMCPPlanner2, h_node::POWTreeObsNode{B,A,O}, s::S, depth)

    tree = h_node.tree
    h = h_node.node

    sol = pomcp.solver

    if POMDPs.discount(pomcp.problem)^depth < sol.eps ||
            POMDPs.isterminal(pomcp.problem, s) ||
            depth >= sol.max_depth
        return 0.0
    end

    if sol.enable_action_pw
        total_n = tree.total_n[h]
        if length(tree.tried[h]) <= sol.k_action*total_n^sol.alpha_action
            if h == 1
                a = next_action(pomcp.next_action, pomcp.problem, tree.root_belief, POWTreeObsNode(tree, h))
            else
                a = next_action(pomcp.next_action, pomcp.problem, tree.beliefs[h], POWTreeObsNode(tree, h))
            end
            if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
                push_anode!(tree, h, a,
                            init_N(sol.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(sol.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            sol.check_repeat_act)
            end
            if length(tree.tried[h]) <= 1
                if depth > 0
                    return POMDPs.discount(pomcp.problem)^depth * estimate_value(pomcp.solved_estimate, pomcp.problem, s, POWTreeObsNode(tree, h), depth)
                else
                    return 0.0
                end
            end
        end
    else # run through all the actions
        if isempty(tree.tried[h])
            action_space_iter = POMDPs.iterator(POMDPs.actions(pomcp.problem))
            anode = length(tree.n)
            for a in action_space_iter
                push_anode!(tree, h, a,
                            init_N(sol.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(sol.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            false)
            end

            if depth > 0 # no need for a rollout if this is the root node
                return POMDPs.discount(pomcp.problem)^depth * estimate_value(pomcp.solved_estimate, pomcp.problem, s, POWTreeObsNode(tree, h), depth)
            else
                return 0.0
            end
        end
    end
    total_n = tree.total_n[h]

    best_node = select_best(pomcp.criterion, h_node)
    a = tree.a_labels[best_node]

    sp, o, r = GenerativeModels.generate_sor(pomcp.problem, s, a, sol.rng)

    new_node = false
    if tree.n_a_children[best_node] <= sol.k_observation*(tree.n[best_node]^sol.alpha_observation)
        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            hao = tree.a_child_lookup[(best_node, o)]
        else
            new_node = true
            hao = length(tree.beliefs) + 1
            push!(tree.beliefs, init_belief(pomcp.node_belief_updater, pomcp.problem, s, a, o, sp))
            push!(tree.total_n, 0)
            push!(tree.tried, Int[])
            push!(tree.o_labels, o)

            if sol.check_repeat_obs
                tree.a_child_lookup[(best_node, o)] = hao
            end
            tree.n_a_children[best_node] += 1
        end

        push!(tree.generated[best_node], o=>hao)
    end

    if !new_node
        pair = rand(sol.rng, tree.generated[best_node])
        o = pair.first
        hao = pair.second
        push_weighted!(tree.beliefs[hao], pomcp.node_belief_updater, s, sp)
        sp = rand(sol.rng, tree.beliefs[hao])
        r = POMDPs.reward(pomcp.problem, s, a, sp) # should cache this so the user doesn't have to implement reward
    end

    if r == Inf
        warn("POMCP: +Inf reward. This is not recommended and may cause future errors.")
    end

    R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, POWTreeObsNode(tree, hao), sp, depth+1)

    tree.n[best_node] += 1
    tree.total_n[h] += 1
    if tree.v[best_node] != -Inf
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node]
    end

    return R
end

