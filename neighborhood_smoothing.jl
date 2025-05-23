using Statistics
using SparseArrays

"""
#     Computes the neighborhood smoothing algorithm described in 
#     Zhang, Levina and Zhu (2017) Estimating network edge probabilities by neighbourhood smoothing, Biometrika 
#     https://academic.oup.com/biomet/article/104/4/771/4158787#100991528

#     Codes are translated from Matlab: https://github.com/yzhanghf/NeighborhoodSmoothing/blob/main/NeighborhoodSmoothing.m
#     First for-loop was changed to allow for asymmetric networks
#     This version takes advantage of column-major ordering (which for large matrices is substantially faster in Julia)
#     """

function NeighborhoodSmoothing(A::Union{SparseMatrixCSC, Matrix}; 
                                directed::Bool = false, 
                                direction::Symbol = :columnwise)
    # INPUT:
    # A: adjacency matrix (NxN)
    # OUTPUT:
    # W_hat: estimated probability matrix
    # Note for directed graphs that the transposition-order of the input matrix matters. 
    # "forward steps" on the matrix (in the sense of finding the neighbors via one forward and one backward step)
    # are steps along the row of the matrix (row-elements are outflows from node)

    if directed && direction == :rowwise    # for row-wise comparisons, run operations on the transpose and then transpose output at the end
        A = copy(transpose(A))
    elseif directed && direction != :columnwise
        throw("direction must be :rowwise or :columnwise")
    end

    N = size(A, 1)
    h = sqrt(log(N) / N)

    # Compute dissimilarity measures
    D = zeros(Float64, N, N)     # sparse matrix writing is not threadsafe

    if directed
        A_sq = A' * A / N
    else
        if A != A'
            throw("Input Matrix not symmetric. For directed graphs choose option directed and specify direction of comparison.")
        end
        A_sq = A * A / N
    end

    Threads.@threads for idx in 1:(N * N)   # this loop is effectively a double-loop but using efficient multi-threading
        i = (idx - 1) ÷ N + 1
        j = (idx - 1) % N + 1
        if i != j
            D[i, j] = maximum(abs.(A_sq[:, i] .- A_sq[:, j]))
        end
    end

    Kernel_mat = zeros(Float64, N, N)
    for i in 1:N
        threshold = quantile(D[:, i], h)
        Kernel_mat[:, i] .= D[:, i] .< threshold
    end

    # Normalize each row under L1 norm
    col_sums = sum(Kernel_mat, dims=1) .+ 1e-10
    Kernel_mat = Kernel_mat ./ col_sums

    W_hat = A * Kernel_mat

    if !directed
        return (W_hat + W_hat') / 2
    elseif direction == :columnwise
        return W_hat
    else
        return Matrix(transpose(W_hat))
    end
end