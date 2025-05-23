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

