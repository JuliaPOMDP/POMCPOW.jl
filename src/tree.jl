
immutable POMCPOWTree{B,A,O,RB}
    # action nodes
    n::Vector{Int}
    v::Vector{Float64}
    generated::Vector{Vector{Pair{O,Int}}}
    a_child_lookup::Dict{Tuple{Int,O}, Int} # may not be maintained based on solver params
    a_labels::Vector{A}
    n_a_children::Vector{Int}

    # observation nodes
    beliefs::Vector{B} # first element is #undef
    total_n::Vector{Int}
    tried::Vector{Vector{Int}} # when we have dpw this will need to be changed to Vector{A}
    o_child_lookup::Dict{Tuple{Int,A}, Int} # may not be maintained based on solver params

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
end

immutable POWTreeObsNode{B,A,O,RB} <: BeliefNode{B,A,O}
    tree::POMCPOWTree{B,A,O,RB}
    node::Int
end

#=
type POWActNode{A,O,BNodeType} <: AbstractActNode
    label::A
    N::Int
    V::Float64
    n_children::Int
    generated::Vector{O}
    children::Dict{O,BNodeType}
end

type POWObsNode{Belief,A,O} <: BeliefNode{Belief,A,O}
    label::O
    N::Int
    B::Belief
    children::Dict{A,POWActNode{A,O,POWObsNode{Belief,A,O}}}
end
=#
