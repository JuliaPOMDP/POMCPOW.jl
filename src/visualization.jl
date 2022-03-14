function D3Trees.D3Tree(p::POMCPOWPlanner; title="POMCPOW Tree", kwargs...)
    @warn("""
         D3Tree(planner::POMCPOWPlanner) is deprecated and may be removed in the future. Instead, please use

             a, info = action_info(planner, state)
             D3Tree(info[:tree])

         Or, you can get this info from a POMDPSimulators History

             info = first(ainfo_hist(hist))
             D3Tree(info[:tree])
         """)

    if p.tree == nothing
        error("POMCPOWPlanner has not constructed a tree yet, run `action(planner, belief)` first to construct the tree.")
    end
    return D3Tree(p.tree; title=title, kwargs...)
end

function D3Trees.D3Tree(t::POMCPOWTree; title="POMCPOW Tree", kwargs...)
    lenb = length(t.total_n)
    lenba = length(t.n)
    len = lenb + lenba
    children = Vector{Vector{Int}}(undef, len)
    text = Vector{String}(undef, len)
    tt = fill("", len)
    link_style = fill("", len)
    style = fill("", len)
    min_V = minimum(t.v)
    max_V = maximum(t.v)
    for b in 1:lenb
        children[b] = t.tried[b] .+ lenb
        text[b] = @sprintf("""
                           o: %s
                           N: %-10d
                           """,
                           b==1 ? "<root>" : node_tag(t.o_labels[b]),
                           t.total_n[b]
                          )
        tt[b] = """
                o: $(b==1 ? "<root>" : node_tag(t.o_labels[b]))
                N: $(t.total_n[b])
                $(length(t.tried[b])) children
                """
        link_width = max(1.0, 20.0*sqrt(t.total_n[b]/t.total_n[1]))
        link_style[b] = "stroke-width:$link_width"
    end
    for ba in 1:lenba
        children[ba+lenb] = collect(unique(last(oipair) for oipair in t.generated[ba]))
        text[ba+lenb] = @sprintf("""
                                 a: %s
                                 N: %-7d V: %-10.3g""",
                                 node_tag(t.a_labels[ba]), t.n[ba], t.v[ba])
        tt[ba+lenb] = """
                      a: $(tooltip_tag(t.a_labels[ba]))
                      N: $(t.n[ba])
                      V: $(t.v[ba])
                      $(length(unique(t.generated[ba]))) children
                      """
        link_width = max(1.0, 20.0*sqrt(t.n[ba]/t.total_n[1]))
        link_style[ba+lenb] = "stroke-width:$link_width"
        rel_V = (t.v[ba]-min_V)/(max_V-min_V)
        if isnan(rel_V)
            color = colorant"gray"
        else
            color = weighted_color_mean(rel_V, colorant"green", colorant"red")
        end
        style[ba+lenb] = "stroke:#$(hex(color))"
    end
    return D3Tree(children;
                  text=text,
                  tooltip=tt,
                  style=style,
                  link_style=link_style,
                  title=title,
                  kwargs...
                 )

end

Base.show(io::IO, mime::MIME"text/html", t::POMCPOWTree) = show(io, mime, D3Tree(t))
Base.show(io::IO, mime::MIME"text/plain", t::POMCPOWTree) = show(io, mime, D3Tree(t))
