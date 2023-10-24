t = CategoricalVector(1, 1.0)
insert!(t, 2, 3.0)
rand(Random.default_rng(), t)

# results = Int[]
# @time for i in 1:1000
#     push!(results, rand(Random.default_rng(), t))
# end
# 
# @time for i in 3:1000
#     insert!(t,i,1.0)
# end
# 
# @time for i in 1:1000
#     push!(results, rand(Random.default_rng(), t))
# end

#=
using Plots
histogram(results)
gui()
=#

t2 = CategoricalVector(2, 1.0)
insert!(t2, 2.0, 3) # test that types can get converted correctly
