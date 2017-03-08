using POMCPOW

t = CategoricalTree(1, 1.0)
insert!(t, 2, 3.0)
rand(Base.GLOBAL_RNG, t)

results = Int[]
@time for i in 1:1000
    push!(results, rand(Base.GLOBAL_RNG, t))
end

@time for i in 3:100
    insert!(t,i,1.0)
end

@time for i in 1:1000
    push!(results, rand(Base.GLOBAL_RNG, t))
end

using Plots
histogram(results)
gui()
