
immutable POMCPOWTree{B,A,O}
    # action nodes
    n::Vector{Int}
    v::Vector{Float64}
    generated::Vector{Vector{Pair{O,Int}}}
    a_child_lookup::Dict{Tuple{Int,O}, Int}
    a_labels::Vector{A}
    n_a_children::Vector{Int}

    # observation nodes
    beliefs::Vector{B} # first element is #undef
    total_n::Vector{Int}
    tried::Vector{Vector{Int}} # when we have dpw this will need to be changed to Vector{A}
    o_child_lookup::Dict{Tuple{Int,A}, Int}

    # root
    root_belief::Any

    POMCPOWTree(root_belief, sz::Int=1000) = new(sizehint!(Int[], sz),
                                                 sizehint!(Int[], sz),
                                                 sizehint!(Vector{Pair{O,Int}}[], sz),
                                                 sizehint!(Dict{Tuple{Int,O}, Int}(), sz),
                                                 sizehint!(A[], sz),
                                                 sizehint!(Int[], sz),

                                                 sizehint!(Array(B,1), sz),
                                                 sizehint!(Int[], sz),
                                                 sizehint!(Vector{Int}[], sz),
                                                 sizehint!(Dict{Tuple{Int,A}, Int}(), sz),
                                                 root_belief)
end

immutable POWTreeObsNode{B,A,O} <: BeliefNode{B,A,O}
    tree::POMCPOWTree{B,A,O}
    node::Int
end

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
