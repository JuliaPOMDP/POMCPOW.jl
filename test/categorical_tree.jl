using POMCPOW

t = CategoricalTree(1, 1.0)
insert!(t, 2, 3.0)
rand(t, Base.GLOBAL_RNG)

results = Int[]
@time for i in 1:1000
    push!(results, rand(t, Base.GLOBAL_RNG))
end

@time for i in 3:10
    insert!(t,i,1.0)
end

@time for i in 1:1000
    push!(results, rand(t, Base.GLOBAL_RNG))
end

using Plots
histogram(results)
gui()
