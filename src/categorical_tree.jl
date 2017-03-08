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

function insert!{T}(t::CategoricalTree{T}, item::T, weight::Float64)
    if t.isleaf
        t.isleaf = false
        t.crit = weight
        t.lchild = CategoricalTree{T}(item, weight)
        t.rchild = CategoricalTree{T}(t.item, t.total)
        t.total += weight
    elseif t.crit < 0.5*t.total
        insert!(t.lchild, item, weight)
        t.crit += weight
        t.total += weight
    else
        insert!(t.rchild, item, weight)
        t.total += weight
    end
end


rand(rng::AbstractRNG, t::CategoricalTree) = rand(t.total*rand(rng), t)

function rand(r::Float64, t::CategoricalTree) # r is a number between 0 and t.total
    if t.isleaf
        return t.item
    elseif r < t.crit
        return rand(r, t.lchild)
    else
        return rand(r-t.crit, t.rchild)
    end
end
