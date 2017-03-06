
type POWActNode{A,O,BNodeType} <: AbstractActNode
    label::A
    N::Int
    V::Float64
    n_children::Int
    generated::Vector{O}
    children::IDict{O,BNodeType}
end

type POWObsNode{Belief,A,O} <: BeliefNode{Belief,A,O}
    label::O
    N::Int
    B::Belief
    children::IDict{A,POWActNode{A,O,POWObsNode{Belief,A,O}}}
end

type IRootNode{RootBelief,A,ANodeType} <: BeliefNode{RootBelief}
    N::Int
    B::RootBelief
    children::IDict{A,ANodeType}
end
