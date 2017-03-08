function simulate{S,A,O,P}(pomcp::POMCPPlanner2{S,A,O,P}, h::Int, s::S, depth)

    tree = get(pomcp.tree)

    sol = pomcp.solver

    if POMDPs.discount(pomcp.problem)^depth < sol.eps ||
            POMDPs.isterminal(pomcp.problem, s) ||
            depth >= sol.max_depth
        return 0.0
    end

    if sol.enable_action_pw
        error("action pw not implemented")
    else # run through all the actions
        if length(tree.total_n) < length(tree.beliefs)
            @assert length(tree.beliefs) - length(tree.total_n) == 1
            action_space_iter = POMDPs.iterator(POMDPs.actions(pomcp.problem))
            anode = length(tree.n)
            total_n = 0
            tried = Int[]
            for a in action_space_iter
                anode += 1
                n = init_N(sol.init_N, pomcp.problem, POWTreeObsNode(tree, h), a)
                push!(tree.n, n)
                push!(tree.v, init_V(sol.init_V, pomcp.problem, POWTreeObsNode(tree, h), a))
                push!(tree.generated, O[])
                push!(tree.a_labels, a)
                push!(tree.n_a_children, 0)
                # tree.o_child_lookup[(h, a)] = anode
                push!(tried, anode)
                total_n += n
            end
            push!(tree.tried, tried)
            push!(tree.total_n, total_n)

            if depth > 0 # no need for a rollout if this is the root node
                return POMDPs.discount(pomcp.problem)^depth * estimate_value(pomcp.solved_estimate, pomcp.problem, s, POWTreeObsNode(tree, h), depth)::Float64
            else
                return 0.0
            end
        end
    end
    total_n = tree.total_n[h]

    # Calculate UCT
    best_criterion_val = -Inf
    local best_node
    for node in tree.tried[h]
        n = tree.n[node]
        if n == 0 && total_n <= 1
            criterion_value = tree.v[node]
        elseif n == 0 && tree.v[node] == -Inf
            criterion_value = Inf
        else
            criterion_value = tree.v[node] + sol.c*sqrt(log(total_n)/n)
        end
        if criterion_value >= best_criterion_val
            best_criterion_val = criterion_value
            best_node = node
        end
    end
    a = tree.a_labels[best_node]

    sp, o, r = GenerativeModels.generate_sor(pomcp.problem, s, a, sol.rng)

    if tree.n_a_children[best_node] <= sol.k_observation*(tree.n[best_node]^sol.alpha_observation)

        push!(tree.generated[best_node], o)

        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            hao = tree.a_child_lookup[(best_node, o)]
        else
            hao = length(tree.beliefs) + 1
            push!(tree.beliefs, POWNodeBelief{S,A,O}(pomcp.problem, s, a, o, sp))

            tree.a_child_lookup[(best_node, o)] = hao
            tree.n_a_children[best_node] += 1
        end

    else
        o = rand(sol.rng, tree.generated[best_node])
        hao = tree.a_child_lookup[(best_node, o)]
        push_weighted!(tree.beliefs[hao], sp)
        sp = rand(sol.rng, tree.beliefs[hao])
        r = POMDPs.reward(pomcp.problem, s, a, sp) # should cache this so the user doesn't have to implement reward
    end

    if r == Inf
        warn("POMCP: +Inf reward. This is not recommended and may cause future errors.")
    end

    R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, hao, sp, depth+1)

    tree.n[best_node] += 1
    tree.total_n[h] += 1
    if tree.v[best_node] != -Inf
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node]
    end

    return R::Float64
end

