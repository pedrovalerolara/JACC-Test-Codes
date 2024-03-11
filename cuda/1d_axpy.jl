import Pkg
Pkg.activate(@__DIR__)


using CUDA
using JACC


function axpy_cuda_kernel(alpha,x,y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds x[i] = x[i] + alpha * y[i]
    return nothing
end

function axpy_cuda(SIZE,alpha,x,y)
    maxPossibleThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE/threads)
    CUDA.@sync @cuda threads = threads blocks = blocks axpy_cuda_kernel(alpha,x,y)
end


println("CUDA:")

SIZE = 100000000
x = ones(SIZE)
y = ones(SIZE)
alpha = 2.0
dx = CuArray(x)
dy = CuArray(y)

# warmup
axpy_cuda(10,alpha,dx,dy)

for i in [10,100,1000,10000,100000,1000000,10000000,100000000]
    @time begin
        axpy_cuda(i, alpha, dx, dy)
    end
end


println("JACC")

function axpy(i, alpha, x, y)
    @inbounds x[i] += alpha * y[i]
end

x = ones(SIZE)
y = ones(SIZE)
jx = JACC.Array(x)
jy = JACC.Array(y)

# warmup
JACC.parallel_for(10, axpy, alpha, jx, jy)

for i in [10,100,1000,10000,100000,1000000,10000000,100000000]
    @time begin
        JACC.parallel_for(i, axpy, alpha, jx, jy)
    end
end

