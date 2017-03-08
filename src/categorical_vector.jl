type CategoricalVector{T}
    items::Vector{T}
    weights::Vector{Float64}
    weight_sum::Float64

    CategoricalVector(item, weight) = new(T[item], Float64[weight], weight)
end

CategoricalVector{T}(item::T, weight::Float64) = CategoricalVector{T}(T[item], Float64[weight], weight)

function insert!{T}(t::CategoricalVector{T}, item::T, weight::Float64)
    push!(t.items, item)
    push!(t.weights, weight)
    t.weight_sum += weight
end

function rand(rng::AbstractRNG, d::CategoricalVector)
    t = rand(rng) * d.weight_sum
    i = 1
    cw = d.weights[1]
    while cw < t && i < length(d.weights)
        i += 1
        @inbounds cw += d.weights[i]
    end
    return d.items[i]
end
