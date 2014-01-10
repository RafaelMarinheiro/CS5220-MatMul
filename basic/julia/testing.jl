# Test script for parallel Julia

using ClusterManagers
addprocs_htc(2)

r = @spawn rand(2,2)
s = @spawn 1+fetch(r)
println(s)
println(fetch(s))
