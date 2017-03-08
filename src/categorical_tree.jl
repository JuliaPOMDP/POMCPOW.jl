#=
type CategoricalTree{T}
    isleaf::Vector{Bool}
    items::Vector{T}
    totals::Vector{Float64}
    crits::Vector{Float64}
    lchildren::Vector{Int}
    rchildren::Vector{Int}

    function CategoricalTree(item, weight)
        new(Bool[true],
            T[item],
            Float64[weight],
            Float64[],
            Int[],
            Int[]
           )
    end
end

CategoricalTree{T}(item::T, weight) = CategoricalTree{T}(item, weight)

function insert!{T}(t::CategoricalTree{T}, item::T, weight::Float64)
    n = 1
    try 
        while !t.isleaf[n]
            if t.crits[n] < 0.5*t.totals[n]
                t.crits[n] += weight
                t.totals[n] += weight
                n = t.lchildren[n]
            else
                t.totals[n] += weight
                n = t.rchildren[n]
            end
        end
    catch ex
        @show t
        rethrow(ex)
    end
    t.isleaf[n] = false
    l = length(t.isleaf)
    resize!(t.crits,l+2)[n] = weight
    resize!(t.lchildren, l+2)[n] = l+1
    push!(t.totals, weight)
    push!(t.items, item)
    push!(t.isleaf, true)
    resize!(t.rchildren, l+2)[n] = l+2
    push!(t.totals, t.totals[n])
    push!(t.items, t.items[n])
    push!(t.isleaf, true)
    t.totals[n] += weight
end


rand(rng::AbstractRNG, t::CategoricalTree) = rand(t.totals[1]*rand(rng), t)

function rand(r::Float64, t::CategoricalTree) # r is a number between 0 and t.totals[1]
    n = 1
    while !t.isleaf[n]
        if r < t.crits[n]
            n = t.lchildren[n]
        else
            r -= t.crits[n]
            n = t.rchildren[n]
        end
    end
    return t.items[n]
end
=#

type CategoricalTree{T}
    isleaf::Bool
    item::T
    total::Float64
    crit::Float64
    lchild::CategoricalTree{T}
    rchild::CategoricalTree{T}

    CategoricalTree(item, weight) = new(true, item, weight) # everything else is #undef
end

CategoricalTree{T}(item::T, weight) = CategoricalTree{T}(item, weight)

function insert!{T}(ct::CategoricalTree{T}, item::T, weight::Float64)
    t = ct
    while !t.isleaf
        if t.crit < 0.5*t.total
            t.crit += weight
            t.total += weight
            t = t.lchild
        else
            t.total += weight
            t = t.rchild
        end
    end
    t.isleaf = false
    t.crit = weight
    t.lchild = CategoricalTree{T}(item, weight)
    t.rchild = CategoricalTree{T}(t.item, t.total)
    t.total += weight
end


rand(rng::AbstractRNG, t::CategoricalTree) = rand(t.total*rand(rng), t)

function rand(r::Float64, ct::CategoricalTree) # r is a number between 0 and t.total
    t = ct
    while !t.isleaf
        if r < t.crit
            t = t.lchild
        else
            r -= t.crit
            t = t.rchild
        end
    end
    return t.item
end
