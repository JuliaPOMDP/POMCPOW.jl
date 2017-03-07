
immutable POMCPOWTree{B,A,O}
    # action nodes
    n::Vector{Int}
    v::Vector{Int}
    generated::Vector{Vector{O}}
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

    POMCPOWTree(root_belief) = new(Int[], Int[], Vector{O}[], Dict{Tuple{Int,O}, Int}(), A[], Int[],
                        Array(B,1), Int[], Vector{Int}[], Dict{Tuple{Int,A}, Int}(),
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
