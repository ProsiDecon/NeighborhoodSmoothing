module WithinGraphKernels

# a module to compute PSD Kernels within a graph, i.e. capturing the similarity of nodes based on their position in the graph.

using LinearAlgebra
using Arpack
using SparseArrays
using Statistics

export neighborhood_smoothing
export gram_matrix



include("neighborhood_smoothing.jl")        # Zhang, Levina and Zhu's Neighborhood Smoothing Algorithm
include("GramMaker.jl")                     # function to compute the Gram Matrix of a graph for different kernels

end