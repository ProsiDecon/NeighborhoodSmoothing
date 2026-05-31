using LinearAlgebra
using Random
using SparseArrays
using Test

include("main.jl")
using .WithinGraphKernels

const WGK = WithinGraphKernels

Random.seed!(20260531)

const ISSUES = String[]

function check(name::AbstractString, f::Function)
    try
        f()
        println("PASS: ", name)
    catch err
        msg = sprint(showerror, err)
        push!(ISSUES, "$name: $msg")
        println("FAIL: ", name, " -- ", msg)
    end
end

function check_throws(name::AbstractString, f::Function)
    try
        f()
        push!(ISSUES, "$name: expected an error, but none was thrown")
        println("FAIL: ", name, " -- expected an error")
    catch
        println("PASS: ", name)
    end
end

function top_eigen_gram(A::AbstractMatrix, k::Int)
    N = size(A, 1)
    vals, vecs = eigen(Matrix(A))
    idx = sortperm(vals; by=abs, rev=true)[1:k]
    return vecs[:, idx] * Diagonal(vals[idx] .^ 2) * vecs[:, idx]' / N
end

function top_svd_column_gram(A::AbstractMatrix, k::Int)
    N = size(A, 1)
    decomp = svd(Matrix(A))
    idx = sortperm(decomp.S; by=abs, rev=true)[1:k]
    V = decomp.V[:, idx]
    return V * Diagonal(decomp.S[idx] .^ 2) * V' / N
end

function top_svd_row_gram(A::AbstractMatrix, k::Int)
    N = size(A, 1)
    decomp = svd(Matrix(A))
    idx = sortperm(decomp.S; by=abs, rev=true)[1:k]
    U = decomp.U[:, idx]
    return U * Diagonal(decomp.S[idx] .^ 2) * U' / N
end

function top_svd_both_gram(A::AbstractMatrix, k::Int)
    return (top_svd_row_gram(A, k) + top_svd_column_gram(A, k)) / 2
end

function random_undirected_unweighted(n::Int)
    A = rand(0:1, n, n)
    A = max.(A, A')
    A[diagind(A)] .= 0
    return Float64.(A)
end

function random_directed_unweighted(n::Int)
    A = rand(0:1, n, n)
    A[diagind(A)] .= 0
    return Float64.(A)
end

function random_undirected_weighted(n::Int)
    A = rand(n, n)
    A = (A + A') / 2
    A[diagind(A)] .= 0
    return A
end

function random_directed_weighted(n::Int)
    A = rand(n, n)
    A[diagind(A)] .= 0
    return A
end

dense_undir_unweighted = random_undirected_unweighted(6)
dense_dir_unweighted = random_directed_unweighted(6)
dense_undir_weighted = random_undirected_weighted(6)
dense_dir_weighted = random_directed_weighted(6)
sparse_undir_unweighted = sparse(random_undirected_unweighted(6))
sparse_dir_weighted = sparse(random_directed_weighted(6))

println("\nPublic gram_matrix smoke tests")

check("dense undirected unweighted adjacency full", () -> begin
    G = gram_matrix(dense_undir_unweighted; type=:adjacency)
    @test G.gram ≈ dense_undir_unweighted * dense_undir_unweighted
    @test G.fitted
end)

check("dense undirected weighted adjacency full", () -> begin
    G = gram_matrix(dense_undir_weighted; type=:adjacency)
    @test G.gram ≈ dense_undir_weighted * dense_undir_weighted
end)

check("sparse undirected unweighted adjacency full", () -> begin
    G = gram_matrix(sparse_undir_unweighted; type=:adjacency)
    @test G.gram ≈ Matrix(sparse_undir_unweighted * sparse_undir_unweighted)
end)

check("dense directed unweighted columnwise full", () -> begin
    G = gram_matrix(dense_dir_unweighted; directed=true, symmetrize=false, direction=:columnwise)
    @test G.gram ≈ dense_dir_unweighted' * dense_dir_unweighted
end)

check("dense directed unweighted rowwise full", () -> begin
    G = gram_matrix(dense_dir_unweighted; directed=true, symmetrize=false, direction=:rowwise)
    @test G.gram ≈ dense_dir_unweighted * dense_dir_unweighted'
end)

check("dense directed weighted both full", () -> begin
    A = dense_dir_weighted
    G = gram_matrix(A; directed=true, symmetrize=false, direction=:both)
    @test G.gram ≈ (A' * A + A * A') / 2
end)

check("sparse directed weighted columnwise full", () -> begin
    A = sparse_dir_weighted
    G = gram_matrix(A; directed=true, symmetrize=false, direction=:columnwise)
    @test G.gram ≈ Matrix(A' * A)
end)

check("dense directed unweighted symmetrized", () -> begin
    A = dense_dir_unweighted
    Asym = Float64.((A + A') .> 0)
    G = gram_matrix(A; directed=true, symmetrize=true)
    @test G.gram ≈ Asym * Asym
end)

check("dense directed weighted symmetrized", () -> begin
    A = dense_dir_weighted
    Asym = (A + A') / 2
    G = gram_matrix(A; directed=true, symmetrize=true)
    @test G.gram ≈ Asym * Asym
end)

check("laplacian full Gram", () -> begin
    A = copy(dense_undir_unweighted)
    L = Diagonal(vec(sum(A; dims=2))) - A
    G = gram_matrix(A; type=:laplacian)
    @test G.gram ≈ L * L
end)

check("neighborhood smoothing kernel smoke", () -> begin
    A = copy(dense_undir_unweighted)
    G = gram_matrix(A; type=:neighborhood_smoothing)
    @test size(G.gram) == size(A)
    @test all(isfinite, G.gram)
end)

check_throws("normalized laplacian intentionally unimplemented", () -> begin
    gram_matrix(dense_undir_unweighted; type=:normalized_laplacian)
end)

println("\nTruncation correctness tests")

small_sym = [0.0 2.0 0.5 0.0;
             2.0 0.0 1.0 0.2;
             0.5 1.0 0.0 1.5;
             0.0 0.2 1.5 0.0]

small_dir = [0.0 1.0 2.0 0.0;
             0.3 0.0 0.0 1.5;
             1.0 0.2 0.0 0.7;
             0.0 1.0 0.4 0.0]

k = 2

check("spectral_truncation! dense vs direct eigendecomposition", () -> begin
    M = zeros(size(small_sym))
    WGK.spectral_truncation!(M, small_sym, k)
    @test M ≈ top_eigen_gram(small_sym, k)
end)

check("spectral_truncation! adjoint vs direct eigendecomposition", () -> begin
    M = zeros(size(small_sym))
    WGK.spectral_truncation!(M, small_sym', k)
    @test M ≈ top_eigen_gram(small_sym', k)
end)

check("spectral_truncation! sparse vs direct eigendecomposition", () -> begin
    M = zeros(size(small_sym))
    WGK.spectral_truncation!(M, sparse(small_sym), k)
    @test M ≈ top_eigen_gram(small_sym, k)
end)

check("singular_truncation! dense columnwise vs direct SVD", () -> begin
    M = zeros(size(small_dir))
    WGK.singular_truncation!(M, small_dir, k)
    @test M ≈ top_svd_column_gram(small_dir, k)
end)

check("singular_truncation! adjoint rowwise convention vs direct SVD", () -> begin
    M = zeros(size(small_dir))
    WGK.singular_truncation!(M, small_dir', k)
    @test M ≈ top_svd_row_gram(small_dir, k)
end)

check("singular_truncation! sparse columnwise vs direct SVD", () -> begin
    M = zeros(size(small_dir))
    WGK.singular_truncation!(M, sparse(small_dir), k)
    @test M ≈ top_svd_column_gram(small_dir, k)
end)

check("singular_truncation_both! dense vs direct SVD", () -> begin
    M = zeros(size(small_dir))
    WGK.singular_truncation_both!(M, small_dir, k)
    @test M ≈ top_svd_both_gram(small_dir, k)
end)

check("singular_truncation_both! adjoint vs direct SVD", () -> begin
    M = zeros(size(small_dir))
    WGK.singular_truncation_both!(M, small_dir', k)
    @test M ≈ top_svd_both_gram(small_dir', k)
end)

check("singular_truncation_both! sparse vs direct SVD", () -> begin
    M = zeros(size(small_dir))
    WGK.singular_truncation_both!(M, sparse(small_dir), k)
    @test M ≈ top_svd_both_gram(small_dir, k)
end)

println("\nPipeline consistency checks")

check("gram_matrix undirected k truncation matches spectral_truncation!", () -> begin
    G = gram_matrix(small_sym; k=k)
    expected = top_eigen_gram(small_sym, k)
    @test G.gram ≈ expected
end)

check("gram_matrix directed columnwise k truncation matches SVD", () -> begin
    G = gram_matrix(small_dir; directed=true, symmetrize=false, direction=:columnwise, k=k)
    @test G.gram ≈ top_svd_column_gram(small_dir, k)
end)

check("gram_matrix directed rowwise k truncation matches SVD", () -> begin
    G = gram_matrix(small_dir; directed=true, symmetrize=false, direction=:rowwise, k=k)
    @test G.gram ≈ top_svd_row_gram(small_dir, k)
end)

check("gram_matrix directed both k truncation matches SVD", () -> begin
    G = gram_matrix(small_dir; directed=true, symmetrize=false, direction=:both, k=k)
    @test G.gram ≈ top_svd_both_gram(small_dir, k)
end)

println("\nMutation checks")

check("adjacency Gram should not mutate input", () -> begin
    A = copy(dense_undir_unweighted)
    before = copy(A)
    gram_matrix(A)
    @test A == before
end)

check("laplacian Gram should not mutate input", () -> begin
    A = copy(dense_undir_unweighted)
    before = copy(A)
    gram_matrix(A; type=:laplacian)
    @test A == before
end)

println("\nSummary")
if isempty(ISSUES)
    println("All checks passed.")
else
    println(length(ISSUES), " issue(s) found:")
    for issue in ISSUES
        println(" - ", issue)
    end
end
