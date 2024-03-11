import Pkg
Pkg.activate(@__DIR__)


using CUDA
using JACC


function dot_cuda_kernel(SIZE, ret, x, y)
    shared_mem = @cuDynamicSharedMem(Float64, 512)
    i = ( blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x
    tmp::Float64 = 0.0
    shared_mem[threadIdx().x] = 0.0

    if i <= SIZE
        tmp = @inbounds x[i] * y[i]
        shared_mem[threadIdx().x] = tmp
    end
    sync_threads()
    if (ti <= 256)
        shared_mem[ti] += shared_mem[ti+256]
    end
    sync_threads()
    if (ti <= 128)
        shared_mem[ti] += shared_mem[ti+128]
    end
    sync_threads()
    if (ti <= 64)
        shared_mem[ti] += shared_mem[ti+64]
    end
    sync_threads()
    if (ti <= 32)
        shared_mem[ti] += shared_mem[ti+32]
    end
    sync_threads()
    if (ti <= 16)
        shared_mem[ti] += shared_mem[ti+16]
    end
    sync_threads()
    if (ti <= 8)
        shared_mem[ti] += shared_mem[ti+8]
    end
    sync_threads()
    if (ti <= 4)
        shared_mem[ti] += shared_mem[ti+4]
    end
    sync_threads()
    if (ti <= 2)
        shared_mem[ti] += shared_mem[ti+2]
    end
    sync_threads()
    if (ti == 1)
        shared_mem[ti] += shared_mem[ti+1]
        ret[blockIdx().x] = shared_mem[ti]
    end
    return nothing
end

function reduce_kernel(SIZE, red, ret)
    shared_mem = @cuDynamicSharedMem(Float64, 512)
    i = ( blockIdx().x - 1) * blockDim().x + threadIdx().x
    ii = i
    tmp::Float64 = 0.0
    if SIZE > 512
        while ii <= SIZE
            tmp += @inbounds red[ii]
            ii += 512
        end
    else
        tmp = @inbounds red[i]
    end
    shared_mem[i] = tmp
    sync_threads()
    if (i <= 256)
        shared_mem[i] += shared_mem[i+256]
    end
    sync_threads()
    if (i <= 128)
        shared_mem[i] += shared_mem[i+128]
    end
    sync_threads()
    if (i <= 64)
        shared_mem[i] += shared_mem[i+64]
    end
    sync_threads()
    if (i <= 32)
        shared_mem[i] += shared_mem[i+32]
    end
    sync_threads()
    if (i <= 16)
        shared_mem[i] += shared_mem[i+16]
    end
    sync_threads()
    if (i <= 8)
        shared_mem[i] += shared_mem[i+8]
    end
    sync_threads()
    if (i <= 4)
        shared_mem[i] += shared_mem[i+4]
    end
    sync_threads()
    if (i <= 2)
        shared_mem[i] += shared_mem[i+2]
    end
    sync_threads()
    if (i == 1)
        shared_mem[i] += shared_mem[i+1]
        ret[1] = shared_mem[1]
    end
    return nothing
end

function dot_cuda(SIZE, x, y)
    maxPossibleThreads = 512
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE/threads)
    ret = CUDA.zeros(Float64,blocks)
    rret = CUDA.zeros(Float64,1)
    CUDA.@sync @cuda threads=threads blocks=blocks shmem = 512 * sizeof(Float64) dot_cuda_kernel(SIZE, ret, x, y)
    CUDA.@sync @cuda threads=threads blocks=1 shmem = 512 * sizeof(Float64) reduce_kernel(blocks, ret, rret)
    return rret
end


println("CUDA:")

SIZE = 100000000
x = ones(SIZE)
y = ones(SIZE)
dx = CuArray(x)
dy = CuArray(y)

# warmup
dot_cuda(10, dx, dy)

for i in [10,100,1000,10000,100000,1000000,10000000,100000000]
    @time begin
        res = dot_cuda(i, dx, dy)
    end
end


println("JACC")

function dot(i, x, y)
    return @inbounds x[i] * y[i]
end

x = ones(SIZE)
y = ones(SIZE)
jx = JACC.Array(x)
jy = JACC.Array(y)

# warmup
JACC.parallel_reduce(10, dot, jx, jy)

for i in [10,100,1000,10000,100000,1000000,10000000,100000000]
    @time begin
        res = JACC.parallel_reduce(i, dot, jx, jy)
    end
end
