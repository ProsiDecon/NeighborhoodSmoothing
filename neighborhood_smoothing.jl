using Statistics
using SparseArrays
#using NearestNeighbors

"""
#     Computes the neighborhood smoothing algorithm described in 
#     Zhang, Levina and Zhu (2017) Estimating network edge probabilities by neighbourhood smoothing, Biometrika 
#     https://academic.oup.com/biomet/article/104/4/771/4158787#100991528

#     Codes are translated from Matlab: https://github.com/yzhanghf/NeighborhoodSmoothing/blob/main/NeighborhoodSmoothing.m
#     First for-loop was changed to allow for asymmetric networks
#     """

#=
# old row-wise version, inefficient
function NeighborhoodSmoothing_rowwise(A::SparseMatrixCSC; directed::Bool = false)

    # INPUT:
    # A: adjacency matrix (NxN)
    # OUTPUT:
    # W_hat: estimated probability matrix
    # Note for directed graphs that the transposition-order of the input matrix matters. 
    # "forward steps" on the matrix (in the sense of finding the neighbors via one forward and one backward step)
    # are steps along the row of the matrix (row-elements are outflows from node)

    N = size(A, 1)
    h = sqrt(log(N) / N)

    # Compute dissimilarity measures
    D = zeros(N, N) # sparse matrix writing is not threadsafe

    if directed
        A_sq = A * A' / N
    else
        A_sq = A * A / N
    end

    Threads.@threads for idx in 1:(N * N)
        i = (idx - 1) ÷ N + 1
        j = (idx - 1) % N + 1
        if i != j
            D[i, j] = maximum(abs.(A_sq[i, :] .- A_sq[j, :]))
        end
    end

    Kernel_mat = zeros(Float64, N, N)
    for i in 1:N
        threshold = quantile(D[i, :], h)
        Kernel_mat[i, :] .= D[i, :] .< threshold
    end

    # Normalize each row under L1 norm
    row_sums = sum(Kernel_mat, dims=2) .+ 1e-10
    Kernel_mat = Kernel_mat ./ row_sums

    W_hat = Kernel_mat * A
    if directed == false
        W_hat = (W_hat + W_hat') / 2
    end

    return W_hat
end
=#


### column-major ordered version (which for large matrices is substantially faster in Julia)
function NeighborhoodSmoothing(A::Union{SparseMatrixCSC, Matrix}; directed::Bool = false, direction::Symbol = :columnwise)

    # INPUT:
    # A: adjacency matrix (NxN)
    # OUTPUT:
    # W_hat: estimated probability matrix
    # Note for directed graphs that the transposition-order of the input matrix matters. 
    # "forward steps" on the matrix (in the sense of finding the neighbors via one forward and one backward step)
    # are steps along the row of the matrix (row-elements are outflows from node)

    if direction == :rowwise    # for row-wise comparisons, run operations on the transpose and then transpose output at the end
        A = copy(transpose(A))
    elseif direction != :columnwise
        throw("direction must be :rowwise or :columnwise")
    end

    N = size(A, 1)
    h = sqrt(log(N) / N)

    # Compute dissimilarity measures
    D = zeros(N, N)     # sparse matrix writing is not threadsafe

    if directed
        A_sq = A' * A / N
    else
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
        W_hat = (W_hat + W_hat') / 2
    end

    if direction == :columnwise
        return W_hat
    else
        return Matrix(W_hat')
    end
end


