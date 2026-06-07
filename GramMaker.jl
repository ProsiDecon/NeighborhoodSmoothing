mutable struct PSDKernel{TA<:AbstractMatrix}
    gram::Matrix{Float64}
    A::TA
    type::Symbol
    k::Union{Int, Nothing}
    t::Union{Float, Nothing}
    directed::Bool
    direction::Symbol
    fitted::Bool
end

"""
    Initialize a PSDKernel object with the instructions for kernel computation
"""
function psdkernel(A::AbstractMatrix, type::Symbol, k::Union{Int, Nothing}, t::Union{Float, Nothing}, directed::Bool, direction::Symbol)
    N = size(A, 1)
    @assert size(A, 1) == size(A, 2) "A must be a square matrix"

    gram = Matrix{Float64}(I, N, N)   # allocate the gram matrix as identity 

    return PSDKernel(gram, copy(A), type, k, t, directed, direction, false)
end

### functions to generate spectral truncation of the kernel matrix
"""
    Helper function to compute the low-rank approximation of the gram matrix based on the top k eigenvalues / singular values if  
        the dense eigendecomposition / singular value decomposition is computed.
"""
function lowrank_gram!(M::AbstractMatrix, V::AbstractMatrix, weights::AbstractVector, normalizer::Real; add::Bool=false)
    weighted_V = similar(M, size(V, 1), length(weights)) 
    weighted_V .= V .* reshape(weights, 1, :) ./ normalizer
    mul!(M, weighted_V, V', one(eltype(M)), add ? one(eltype(M)) : zero(eltype(M)))
    return M
end

"""
    Computes the spectral truncation according to the first k eigenvalues and  
        returns the inner product (Gram matrix) based on this truncation.
"""
function spectral_truncation!(M::Matrix, A::Matrix, k::Int)
    # compute the top k eigenvalues and eigenvectors of the symmetric matrix M
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"
    
    if k == N
        error("For spectral truncation, specify truncation dimension k < N, the matrix dimension.")
    elseif k >= .1*N   # for large k, it is more efficient to compute the full eigendecomposition and then truncate
        evals, evecs = eigen(A)
        idx = sortperm(evals; by = abs, rev=true)[1:k]
        lowrank_gram!(M, view(evecs, :, idx), evals[idx].^2, N)
        return M
    else
        evals, evecs = eigs(A, nev = k, which = :LM)
        lowrank_gram!(M, evecs, evals.^2, N)
        #M .= evecs * Diagonal(evals.^2) * evecs'
        #M ./= N
        return M
    end
end

"""
    Computes the spectral truncation according to the first k eigenvalues and  
        returns the inner product (Gram matrix) based on this truncation.
"""
function spectral_truncation!(M::Matrix, A::Adjoint, k::Int)
    # compute the top k eigenvalues and eigenvectors of the symmetric matrix M
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        error("For spectral truncation, specify truncation dimension k < N, the matrix dimension.")
    elseif k >= .1*N   # for large k, it is more efficient to compute the full eigendecomposition and then truncate
        evals, evecs = eigen(A)
        idx = sortperm(evals; by = abs, rev=true)[1:k]
        lowrank_gram!(M, view(evecs, :, idx), evals[idx].^2, N)
        return M
    else
        evals, evecs = eigs(A, nev = k, which = :LM)
        lowrank_gram!(M, evecs, evals.^2, N)
        #M .= evecs * Diagonal(evals.^2) * evecs'
        #M ./= N
        return M
    end
end

"""
    Computes the spectral truncation according to the first k eigenvalues and  
        returns the inner product (Gram matrix) based on this truncation.
"""
function spectral_truncation!(M::Matrix, A::SparseMatrixCSC, k::Int)
    # compute the top k eigenvalues and eigenvectors of the symmetric matrix M
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        error("For spectral truncation, specify truncation dimension k < N, the matrix dimension.")
    else
        evals, evecs = eigs(A, nev = k, which = :LM)
        lowrank_gram!(M, evecs, evals.^2, N)
        #M .= evecs * Diagonal(evals.^2) * evecs'
        #M ./= N
        return M
    end
end

"""
    Computes the singular truncation according to the first k singular values and  
        returns the inner product (Gram matrix) based on this truncation in column-direction (i.e. A'A).
"""
function singular_truncation!(M::Matrix, A::Matrix, k::Int)
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        error("For singular truncation, specify truncation dimension k < N, the matrix dimension.")
    elseif k >= .1*N 
        _, S, V = svd(A)
        idx = sortperm(S; by = abs, rev=true)[1:k]
        lowrank_gram!(M, view(V, :, idx), S[idx].^2, N)
        return M
    else
        singulars = svds(A, nsv = k)[1]
        lowrank_gram!(M, singulars.Vt', singulars.S.^2, N)
        #M .= singulars.U * Diagonal(singulars.S.^2) * singulars.U'
        #M ./= N
        return M
    end
end

"""
    Computes the singular truncation according to the first k singular values and  
        returns the inner product (Gram matrix) based on this truncation in column-direction (i.e. A'A).
"""
function singular_truncation!(M::Matrix, A::Adjoint, k::Int)
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        error("For singular truncation, specify truncation dimension k < N, the matrix dimension.")
    elseif k >= .1*N 
        _, S, V = svd(A)
        idx = sortperm(S; by = abs, rev=true)[1:k]
        lowrank_gram!(M, view(V, :, idx), S[idx].^2, N)
        return M
    else
        singulars = svds(A, nsv = k)[1]
        lowrank_gram!(M, singulars.Vt', singulars.S.^2, N)
        #M .= singulars.U * Diagonal(singulars.S.^2) * singulars.U'
        #M ./= N
        return M
    end
end

"""
    Computes the singular truncation according to the first k singular values and  
        returns the inner product (Gram matrix) based on this truncation in column-direction (i.e. A'A).
"""
function singular_truncation!(M::Matrix, A::SparseMatrixCSC, k::Int)
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        error("For singular truncation, specify truncation dimension k < N, the matrix dimension.")
    else
        singulars = svds(A, nsv = k)[1]
        lowrank_gram!(M, singulars.Vt', singulars.S.^2, N)
        #M .= singulars.U * Diagonal(singulars.S.^2) * singulars.U'
        #M ./= N
        return M
    end
end

"""
    Computes the singular truncation according to the first k singular values and  
        returns the inner product (Gram matrix) based on this truncation in both directions (i.e. (A'A + AA')/2).
"""
function singular_truncation_both!(M::Matrix, A::Matrix, k::Int)
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        error("For singular truncation, specify truncation dimension k < N, the matrix dimension.")
    elseif k >= .1*N 
        U, S, V = svd(A)
        idx = sortperm(S; by = abs, rev=true)[1:k]
        lowrank_gram!(M, view(U, :, idx), S[idx].^2, 2*N)
        lowrank_gram!(M, view(V, :, idx), S[idx].^2, 2*N; add=true) # add = true adds the object to existing M
        return M
    else
        singulars = svds(A, nsv = k)[1]
        lowrank_gram!(M, singulars.U, singulars.S.^2, 2*N)
        lowrank_gram!(M, singulars.Vt', singulars.S.^2, 2*N; add = true)
        #M .= singulars.U * Diagonal(singulars.S.^2) * singulars.U' + singulars.Vt' * Diagonal(singulars.S.^2) * singulars.Vt
        #M ./= 2*N
        return M
    end
end


"""
    Computes the singular truncation according to the first k singular values and  
        returns the inner product (Gram matrix) based on this truncation in both directions (i.e. (A'A + AA')/2).
"""
function singular_truncation_both!(M::Matrix, A::Adjoint, k::Int)
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        error("For singular truncation, specify truncation dimension k < N, the matrix dimension.")
    elseif k >= .1*N 
        U, S, V = svd(A)
        idx = sortperm(S; by = abs, rev=true)[1:k]
        lowrank_gram!(M, view(U, :, idx), S[idx].^2, 2*N)
        lowrank_gram!(M, view(V, :, idx), S[idx].^2, 2*N; add=true) # add = true adds the object to existing M
        return M
    else
        singulars = svds(A, nsv = k)[1]
        lowrank_gram!(M, singulars.U, singulars.S.^2, 2*N)
        lowrank_gram!(M, singulars.Vt', singulars.S.^2, 2*N; add = true)
        #M .= singulars.U * Diagonal(singulars.S.^2) * singulars.U' + singulars.Vt' * Diagonal(singulars.S.^2) * singulars.Vt
        #M ./= 2*N
        return M
    end
end

"""
    Computes the singular truncation according to the first k singular values and  
        returns the inner product (Gram matrix) based on this truncation in both directions (i.e. (A'A + AA')/2).
"""
function singular_truncation_both!(M::Matrix, A::SparseMatrixCSC, k::Int)
    N = size(M, 1)
    @assert k <= N "k must be less than or equal to the dimension of the matrix"

    if k == N
        error("For singular truncation, specify truncation dimension k < N, the matrix dimension.")
    else
        singulars = svds(A, nsv = k)[1]
        lowrank_gram!(M, singulars.U, singulars.S.^2, 2*N)
        lowrank_gram!(M, singulars.Vt', singulars.S.^2, 2*N; add = true)
        #M .= singulars.U * Diagonal(singulars.S.^2) * singulars.U' + singulars.Vt' * Diagonal(singulars.S.^2) * singulars.Vt
        #M ./= 2*N
        return M
    end
end
    
"""
    Fits the kernel to the input matrix A and initializes the gram matrix.
"""
function fit_initialised!(kernel::PSDKernel)
    N = size(kernel.A,1)

    if !isnothing(kernel.k) && kernel.k >= N
        @warn "k is larger than the dimension of the graph. Returning full gram matrix."
        kernel.k = nothing
    end

    if kernel.type in [:laplacian, :normalized_laplacian] && issymmetric(kernel.A)  # compute the heat kernel
        @assert !isnothing(kernel.t) "For the heat diffusion kernel, specify a diffusion parameter t > 0."
        if typeof(kernel.A) <: SparseMatrixCSC
            kernel.gram .= exp(-kernel.t .* Matrix(kernel.A))  
        elseif typeof(kernel.A) <: Matrix
            kernel.gram .= exp(-kernel.t .* kernel.A)  
        end
    elseif (typeof(kernel.k) <:Int) && (kernel.k < N)
        if !kernel.directed
            spectral_truncation!(kernel.gram, kernel.A, kernel.k)
        elseif kernel.direction == :columnwise
            singular_truncation!(kernel.gram, kernel.A, kernel.k)
        elseif kernel.direction == :rowwise
            singular_truncation!(kernel.gram, kernel.A', kernel.k)
        elseif kernel.direction == :both
            singular_truncation_both!(kernel.gram, kernel.A, kernel.k)
        else
            error("direction must be either :columnwise, :rowwise, or :both")
        end
    elseif isnothing(kernel.k) || N == kernel.k
        if !kernel.directed
            mul!(kernel.gram, kernel.A, kernel.A)   # if no truncation, the gram matrix is just the inner product of the choice of (symmetric) network matrix with itself
        elseif kernel.direction == :columnwise
            mul!(kernel.gram, kernel.A', kernel.A)   # for columnwise, the gram matrix is A'A
        elseif kernel.direction == :rowwise
            mul!(kernel.gram, kernel.A, kernel.A')   # for rowwise, the gram matrix is AA'
        elseif kernel.direction == :both
            mul!(kernel.gram, kernel.A', kernel.A)   # for both, this does A'A and allocates to kernel.gram
            mul!(kernel.gram, kernel.A, kernel.A', 0.5, 0.5)  # this does 0.5 * (A'A) + 0.5 (AA')
        else
            error("direction must be either :columnwise, :rowwise, or :both")
        end
    else
        @assert typeof(kernel.k) == Nothing "k must be either an integer or nothing"
    end
end

function fit!(kernel::PSDKernel) 
    if kernel.fitted
        @warn "Kernel already fitted. Re-fitting will overwrite the existing gram matrix."
    end

    # this block stores in the kernel.gram the matrix of choice underlying the kernel
    if kernel.type == :adjacency                    # computes the Gram matrix on the neighborhood directly
        nothing
    elseif kernel.type == :neighborhood_smoothing   # computes the Gram matrix on Zhang, Levina, and Zhu's graphon estimate
        kernel.direction == :both && @warn "Neighborhood smoothing for both directions not implemented. Graphon will be estimated column-wise."
        smoothing_direction = ifelse(kernel.direction == :both, :columnwise, kernel.direction)
        kernel.A = neighborhood_smoothing(kernel.A; directed = kernel.directed, direction = smoothing_direction, returndist = false)
    elseif kernel.type == :laplacian                   # computes the Gram matrix on the unnormalized graph Laplacian
        # for the directed case, we can initiate two Laplacians. For exposure to upstream flows, we should consider the Laplacian
        # placing outdegrees on the main diagonal and only conduct column-wise operations, see differences to Peebles' definition https://www.youtube.com/watch?v=3j3IRXdrzEU&t=129s
        !kernel.directed || @warn "The Graph Laplacian on a directed graph may not be positive semi-definite. Consider setting symmetrize = true. Continuing will compute a gram matrix on the directed Laplacian and not a heat diffusion kernel."
        !kernel.directed || kernel.direction == :columnwise || @warn "The Graph Laplacian in the directed case is defined as D_out - A. Use column-wise operations only."
        outdeg = vec(sum(kernel.A, dims=1))
        kernel.A .*= -1
        kernel.A += Diagonal(outdeg) 
    elseif kernel.type == :normalized_laplacian        # computes the Gram matrix on the normalized graph Laplacian, 
        # this implementation uses L_rw = I - D^{-1} A  due to the arguments in von Luxburg 2007 A Tutorial on Spectral Clustering
        # Newman (chapter 6.14) states that the Graph Laplacian is not really applicable to directed networks
        !kernel.directed || @warn "The normalised Graph Laplacian on a directed graph may not be positive semi-definite. Consider setting symmetrize = true. Continuing will compute a gram matrix on the directed Laplacian and not a heat diffusion kernel."
        !kernel.directed || kernel.direction == :columnwise || @warn "The Graph Laplacian in the directed case is defined as D_out - A. Use column-wise operations only."
        outdeg = (vec(sum(kernel.A, dims=1)) .+ eps()).^(-1)
        N = length(outdeg)
        if kernel.A isa SparseMatrixCSC
            kernel.A = sparse(I,N,N) - Diagonal(outdeg) * kernel.A
        else
            kernel.A = Matrix(I,N,N) - kernel.A .* outdeg
        end
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
                    t::Union{Float, Nothing} = nothing, # the time parameter for the heat diffusion kernel
                    directed::Bool = false,             # whether the graph is intended to be directed
                    symmetrize::Bool = true,            # for directed graphs, whether to compute the gram matrix of the symmetric graph (true) or whether to use the directed graph
                    direction::Symbol = :columnwise)    # if !symmetrize, the direction of the kernel (forward = :rowwise, backward = :columnwise, computing both and using their average = :both)
    
    type in [:adjacency, :neighborhood_smoothing, :laplacian, :normalized_laplacian] || error("type must be one of :adjacency, :neighborhood_smoothing, :laplacian, :normalized_laplacian")

    if type in [:laplacian, :normalized_laplacian] 
        !isnothing(t) || error("For the heat diffusion kernel, specify a diffusion parameter t > 0.")
    else
        t = nothing
    end

    checksymmetric = issymmetric(A)
    checkunweighted = all(x -> ((x == 0) || (x == 1)), A)
    directed || @assert checksymmetric "A is not symmetric, but directed is set to false."

    if checksymmetric
        !directed || @warn "A is symmetric but directed was set true. Now set to false."
        directed = false
    else
        if symmetrize
            if checkunweighted
                A = ((A + A') .> 0) .* 1
            else
                A += A'
                A ./= 2
            end 
            directed = false
        end
        (!directed || direction in [:columnwise, :rowwise, :both]) || error("direction must be either :columnwise, :rowwise, or :both")
    end

    # initiate the container 
    gram = psdkernel(A, type, k, t, directed, direction)
    fit!(gram)   

    return gram
end

