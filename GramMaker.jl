struct PSDKernel
    gram::AbstractMatrix
    type::Symbol
    k::Union{Int, Nothing}
    directed::Bool
    direction::Symbol
    fitted::Bool
end

"""
    Initialize a PSDKernel object with the instructions for kernel computation
"""
function psdkernel(A::AbstractMatrix, type::Symbol, k::Union{Int, Nothing}, directed::Bool, direction::Symbol)
    N = size(A, 1)
    @assert size(A, 1) == size(A, 2) "A must be a square matrix"

    T = eltype(A)
    
    gram = Matrix{T}(I, N, N)   # allocate the gram matrix as identity so we can later in-place rmul! it with the kernel matrix

    return PSDKernel(gram, type, k, directed, direction, false)
end

### functions to generate spectral truncation of the kernel matrix
"""
    Computes the spectral truncation according to the first k eigenvalues and  
        returns the inner product (Gram matrix) based on this truncation.
"""
function spectral_truncation!(M::Matrix, k::Int)
    # compute the top k eigenvalues and eigenvectors of the symmetric matrix M
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        return M
    elseif k >= .1*N   # for large k, it is more efficient to compute the full eigendecomposition and then truncate
        evals, evecs = eigen(M)
        idx = sortperm(evals, rev=true)[1:k]
        M .= evecs[:, idx] * Diagonal((evals[idx]).^2) * evecs[:, idx]'
        M ./= N
        return M
    else
        evals, evecs = eigs(M, nev = k, which = :LM)
        M .= evecs * Diagonal(evals.^2) * evecs'
        M ./= N
        return M
    end
end

"""
    Computes the spectral truncation according to the first k eigenvalues and  
        returns the inner product (Gram matrix) based on this truncation.
"""
function spectral_truncation!(M::SparseMatrixCSC, k::Int)
    # compute the top k eigenvalues and eigenvectors of the symmetric matrix M
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        return M
    else
        evals, evecs = eigs(M, nev = k, which = :LM)
        M .= evecs * Diagonal(evals.^2) * evecs'
        M ./= N
        return M
    end
end

"""
    Computes the singular truncation according to the first k singular values and  
        returns the inner product (Gram matrix) based on this truncation in column-direction (i.e. A'A).
"""
function singular_truncation!(M::Matrix, k::Int)
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        return M
    else
        U, S, _ = svd(M)
        M .= U[:, 1:k] * Diagonal(S[1:k].^2) * U[:, 1:k]'
        M ./= N
        return M
    end
end

"""
    Computes the singular truncation according to the first k singular values and  
        returns the inner product (Gram matrix) based on this truncation in column-direction (i.e. A'A).
"""
function singular_truncation!(M::SparseMatrixCSC, k::Int)
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        return M
    else
        U, S, _ = svds(M, nev = k, which = :LM)
        M .= U * Diagonal(S.^2) * U'
        M ./= N
        return M
    end
end

"""
    Computes the singular truncation according to the first k singular values and  
        returns the inner product (Gram matrix) based on this truncation in column-direction (i.e. A'A).
"""
function singular_truncation_both!(M::Matrix, k::Int)
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        return M
    else
        U, S, V = svd(M)
        M .= U[:, 1:k] * Diagonal(S[1:k].^2) * U[:, 1:k]' + V[:, 1:k] * Diagonal(S[1:k].^2) * V[:, 1:k]'
        M ./= 2*N
        return M
    end
end

"""
    Computes the singular truncation according to the first k singular values and  
        returns the inner product (Gram matrix) based on this truncation in column-direction (i.e. A'A).
"""
function singular_truncation_both!(M::SparseMatrixCSC, k::Int)
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        return M
    else
        U, S, V = svds(M, nev = k, which = :LM)
        M .= U * Diagonal(S.^2) * U' + V * Diagonal(S.^2) * V'
        M ./= 2*N
        return M
    end
end
    
"""
    Fits the kernel to the input matrix A and initializes the gram matrix.
"""
function fit_initialised!(kernel::PSDKernel)
    if kernel.k >= size(kernel.gram, 1)
        @warn "k is larger than the dimension of the graph. Returning full gram matrix."
        kernel.k = nothing
    end
    if typeof(kernel.k) <:Int
        if !kernel.directed
            spectral_truncation!(kernel.gram, kernel.k)
        elseif kernel.direction == :columnwise
            singular_truncation!(kernel.gram, kernel.k)
        elseif kernel.direction == :rowwise
            singular_truncation!(kernel.gram', kernel.k)
        elseif kernel.direction == :both
            singular_truncation_both!(kernel.gram, kernel.k)
        else
            error("direction must be either :columnwise, :rowwise, or :both")
        end
    elseif isnothing(kernel.k)
        if !kernel.directed
            rmul!(kernel.gram, kernel.gram)   # if no truncation, the gram matrix is just the inner product of the choice of (symmetric) network matrix with itself
        elseif kernel.direction == :columnwise
            lmul!(kernel.gram', kernel.gram)   # for columnwise, the gram matrix is A'A
        elseif kernel.direction == :rowwise
            rmul!(kernel.gram, kernel.gram')   # for rowwise, the gram matrix is AA'
        elseif kernel.direction == :both
            AAt = copy(kernel.gram * kernel.gram')
            lmul!(kernel.gram', kernel.gram)   # for both, the gram matrix is (A'A + AA')/2
            kernel.gram += AAt
            kernel.gram ./= 2
        else
            error("direction must be either :columnwise, :rowwise, or :both")
        end
    else
        @assert typeof(kernel.k) == Nothing "k must be either an integer or nothing"
    end
end

function fit!(kernel::PSDKernel, A::AbstractMatrix) 
    if kernel.fitted
        @warn "Kernel already fitted. Re-fitting will overwrite the existing gram matrix."
        kernel = psdkernel(A, kernel.type, kernel.k, kernel.directed, kernel.direction)   # re-initialize the gram matrix to identity before re-fitting
    end

    # this block stores in the kernel.gram the matrix of choice underlying the kernel
    if kernel.type == :adjacency                    # computes the Gram matrix on the neighborhood directly
        rmul!(kernel.gram, A)
    elseif kernel.type == :neighborhood_smoothing   # computes the Gram matrix on Zhang, Levina, and Zhu's graphon estimate
        P_hat = neighborhood_smoothing(A; directed = kernel.directed, direction = kernel.direction, returndist = false)
        rmul!(kernel.gram, P_hat)
    elseif kernel.type == :laplacian                   # computes the Gram matrix on the unnormalized graph Laplacian
        L = Diagonal(sum(A, dims=2))
        L -= A
        rmul!(kernel.gram, L)
    elseif kernel.type == :normalized_laplacian        # computes the Gram matrix on the normalized graph Laplacian
        @warn "The normalized Laplacian is in beta version."
        D_inv_sqrt = Diagonal(1 ./ sqrt.(sum(A, dims=2)) .+ 1e-10)   # add small constant to avoid division by zero
        L_norm = I - D_inv_sqrt * A * D_inv_sqrt
        rmul!(kernel.gram, L_norm)
    else
        error("type must be one of :adjacency, :neighborhood_smoothing, :laplacian, :normalized_laplacian")
    end

    # based on the stored network-object, and preferences for directed graphs, this block applies the correct spectral/singular truncation and computes the gram matrix in-place
    fit_initialised!(kernel)

    kernel.fitted = true
    return kernel
end

#= note the Gram matrix will (almost) always be dense
function psdkernel(A::SparseMatrixCSC, type::Symbol, k::Union{Int, Nothing}, directed::Bool, symmetrize::Bool, direction::Symbol)
    N = size(A, 1)
    @assert size(A, 1) == size(A, 2) "A must be a square matrix"

    T = eltype(A)
    
    gram = spzeros(T, N, N)

    return PSDKernel(gram, type, k, directed, direction)
end
=#


"""
    gram_matrix(A; type = :adjacency, k = nothing, directed = false, symmetrize = true, direction = :columnwise)

    A::AbstractMatrix - the adjacency matrix of the graph for which to compute the gram matrix
    type::Symbol - the type of kernel for which to compute the gram matrix. One of
        :adjacency - the adjacency matrix itself (or its symmetrized version if symmetrize = true)
        :neighborhood_smoothing - the neighborhood smoothing kernel of Zhang, Levina and Zhu (2022)
        :laplacian - the unnormalized graph Laplacian
        :normalized_laplacian - the normalized graph Laplacian
    k::Union{Int, Nothing} - the number of principal components of which to return the gram matrix (if nothing, use the full information)
    directed::Bool - whether the graph is intended to be directed 
    symmetrize::Bool - for directed graphs, whether to compute the gram matrix of the symmetric graph
    direction::Symbol - if !symmetrize, the direction of the kernel (forward = :rowwise, backward = :columnwise, computing both and using their average = :both)
    
    Computes the Gram matrix of node-similarity within a graph for different choices of Kernel.
        Allows for the choice of spectral truncation of the underlying kernel.
        For unidirected graphs, this is done via spectral decomposition. 
        For directed graphs, it is done via singular value decomposition.
"""
function gram_matrix(A::AbstractMatrix;                  # the adjacency matrix
                    type::Symbol = :adjacency,          # the type of Kernel for which we compute the Gram matrix
                    k::Union{Int, Nothing} = nothing,   # the number of principal components of which to return the gram matrix (if nothing, use the full information)
                    directed::Bool = false,             # whether the graph is intended to be directed
                    symmetrize::Bool = true,            # for directed graphs, whether to compute the gram matrix of the symmetric graph (true) or whether to use the directed graph
                    direction::Symbol = :columnwise)    # if !symmetrize, the direction of the kernel (forward = :rowwise, backward = :columnwise, computing both and using their average = :both)
    
    type in [:adjacency, :neighborhood_smoothing, :laplacian, :normalized_laplacian] || error("type must be one of :adjacency, :neighborhood_smoothing, :laplacian, :normalized_laplacian")

    issymmetric = (A == A')
    @assert !directed || !issymmetric "A is not symmetric, but directed is set to false."
    if issymmetric
        directed = false
    else
        if symmetrize
            A += A'
            A ./= 2
            directed = false
        end
        (!directed || direction in [:columnwise, :rowwise, :both]) || error("direction must be either :columnwise, :rowwise, or :both")
    end

    # initiate the container 
    gram = psdkernel(A, type, k, directed, direction)
    fit!(gram, A)   

    return gram
end

