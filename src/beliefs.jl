immutable POWNodeBelief{S,A,O,P}
    model::P
    a::A
    o::O
    dist::CategoricalTree{S}

    POWNodeBelief(m,a,o,d) = new(m,a,o,d)
    function POWNodeBelief(m::P, s::S, a::A, o::O, sp::S)
        new(m, a, o, CategoricalTree{S}(sp, obs_weight(m, s, a, sp, o)))
    end
end

function POWNodeBelief{S,A,O}(model::POMDP{S,A,O}, s::S, a::A, o::O, sp::S)
    POWNodeBelief{S,A,O,typeof(model)}(model, s, a, o, sp)
end

rand(rng::AbstractRNG, b::POWNodeBelief) = rand(rng, b.dist)
mean(b::POWNodeBelief) = mean(b.dist)


immutable POWNodeFilter end

belief_type{P<:POMDP}(::Type{POWNodeFilter}, ::Type{P}) = POWNodeBelief{state_type(P), action_type(P), obs_type(P), P}
init_node_belief{S,A,O,P<:POMDP}(::POWNodeFilter, p::P, s::S, a::A, o::O, sp::S) = POWNodeBelief(p, s, a, o, sp) 
function push_weighted!(b::POWNodeBelief, ::POWNodeFilter, s, sp)
    w = obs_weight(b.model, s, b.a, sp, b.o)
    insert!(b.dist, sp, w)
end
