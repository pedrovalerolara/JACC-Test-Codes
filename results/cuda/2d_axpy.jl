import Pkg
Pkg.activate(@__DIR__)


using CUDA
using JACC


function axpy_cuda_kernel(alpha,x,y)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  @inbounds x[i,j] = x[i,j] + alpha * y[i,j]
  return nothing
end

function axpy_cuda((M,N),alpha,x,y)
  maxPossibleThreads = 16 
  Mthreads = min(M, maxPossibleThreads)
  Mblocks = ceil(Int, M/Mthreads)
  Nthreads = min(N, maxPossibleThreads)
  Nblocks = ceil(Int, N/Nthreads)
  CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks=(Mblocks, Nblocks) axpy_cuda_kernel(alpha,x,y)
end


println("CUDA:")

SIZE = 10000
x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
alpha = 2.0
dx = CuArray(x)
dy = CuArray(y)

# warmup
axpy_cuda((1000,1000),alpha,dx,dy)

for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
 @time begin
  axpy_cuda((i,i),alpha,dx,dy)
 end
end


function axpy(i, j, alpha, x, y)
    @inbounds x[i,j] = x[i,j] + alpha * y[i,j]
end

println("JACC")

x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
jx = JACC.Array(x)
jy = JACC.Array(y)

# warmup
JACC.parallel_for((1000,1000), axpy, alpha, jx, jy)

for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
 @time begin
  JACC.parallel_for((i,i), axpy, alpha, jx, jy)
 end
end
