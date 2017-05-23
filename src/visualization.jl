import JSON
import MCTS: AbstractTreeVisualizer, node_tag, tooltip_tag, create_json, blink

type POMCPOWVisualizer <: AbstractTreeVisualizer
    tree::POMCPOWTree
end
POMCPOWVisualizer(planner::POMCPPlanner2) = POMCPOWVisualizer(get(planner.tree))

blink(planner::POMCPPlanner2) = blink(POMCPOWVisualizer(get(planner.tree)))

typealias NodeDict Dict{Int, Dict{String, Any}}

function create_json(v::POMCPOWVisualizer)
    t = v.tree
    nb = length(t.sr_beliefs)
    na = length(t.n)
    if nb + na > 10_000
        warn("Creating json for a tree with $(na+nb) nodes - this could take a while")
    end
    nd = NodeDict()
    for id in 1:nb
        if id == 1
            obs = "root"
            tt_tag = tooltip_tag(obs)
        else
            obs = t.o_labels[id]
            tt_tag = "$(tooltip_tag(obs)) [$(n_items(t.sr_beliefs[id].dist)) particles]"
        end
        nd[id] = Dict("id"=>id,
                      "type"=>:obs,
                      "children_ids"=>Int[j+nb for j in t.tried[id]],
                      "tag"=>node_tag(obs),
                      "tt_tag"=>tt_tag,
                      "N"=>t.total_n[id]
                     )
    end
    for id in 1:na
        a = t.a_labels[id]
        nd[id+nb] = Dict("id"=>id+nb,
                      "type"=>:action,
                      "children_ids"=>Int[pair.second for pair in t.generated[id]],
                      "tag"=>node_tag(a),
                      "tt_tag"=>tooltip_tag(a),
                      "N"=>t.n[id],
                      "Q"=>t.v[id]
                     )
    end
    json = JSON.json(nd)
    return (json, 1)
end
