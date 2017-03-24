
immutable POMCPOWTree{B,A,O,RB}
    # action nodes
    n::Vector{Int}
    v::Vector{Float64}
    generated::Vector{Vector{Pair{O,Int}}}
    a_child_lookup::Dict{Tuple{Int,O}, Int} # may not be maintained based on solver params
    a_labels::Vector{A}
    n_a_children::Vector{Int}

    # observation nodes
    sr_beliefs::Vector{B} # first element is #undef
    total_n::Vector{Int}
    tried::Vector{Vector{Int}} # when we have dpw this will need to be changed to Vector{A}
    o_child_lookup::Dict{Tuple{Int,A}, Int} # may not be maintained based on solver params
    o_labels::Vector{O}

    # root
    root_belief::RB

    POMCPOWTree(root_belief, sz::Int=1000) = new(sizehint!(Int[], sz),
                                                 sizehint!(Int[], sz),
                                                 sizehint!(Vector{Pair{O,Int}}[], sz),
                                                 Dict{Tuple{Int,O}, Int}(),
                                                 sizehint!(A[], sz),
                                                 sizehint!(Int[], sz),

                                                 sizehint!(Array(B,1), sz),
                                                 sizehint!(Int[0], sz),
                                                 sizehint!(Vector{Int}[Int[]], sz),
                                                 Dict{Tuple{Int,A}, Int}(),
                                                 sizehint!(Array(O,1), sz),

                                                 root_belief)
end

@inline function push_anode!{B,A,O}(tree::POMCPOWTree{B,A,O}, h::Int, a::A, n::Int=0, v::Float64=0.0, update_lookup=true)
    anode = length(tree.n) + 1
    push!(tree.n, n)
    push!(tree.v, v)
    push!(tree.generated, Pair{O,Int}[])
    push!(tree.a_labels, a)
    push!(tree.n_a_children, 0)
    if update_lookup
        tree.o_child_lookup[(h, a)] = anode
    end
    push!(tree.tried[h], anode)
    tree.total_n[h] += n
    return anode
end

immutable POWTreeObsNode{B,A,O,RB} <: BeliefNode{B,A,O}
    tree::POMCPOWTree{B,A,O,RB}
    node::Int
end

belief(h::POWTreeObsNode) = ifelse(h.node==1, h.tree.root_belief, StateBelief(h.tree.sr_beliefs[h.node]))
n_children(h::POWTreeObsNode) = length(h.tree.tried[h.node])
