module JACCTestCodesCUDA
using CUDA
using JACCTestCodes
import JACCTestCodes: dot, axpy, matvecmul, lbm

# #-------------------------1D AXPY

function axpy_kernel_1d(alpha,x,y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds x[i] = x[i] + alpha * y[i]
    return nothing
end
#-------------------------1D DOT

function axpy_kernel_2d(alpha,x,y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds x[i,j] = x[i,j] + alpha * y[i,j]
    return nothing
end

function JACCTestCodes.axpy(SIZE::Int,alpha,x::CuArray,y::CuArray)
    maxPossibleThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE/threads)
    CUDA.@sync @cuda threads = threads blocks = blocks axpy_kernel_1d(alpha,x,y)
end

function JACCTestCodes.axpy((M,N)::Tuple{Int, Int},alpha,x::CuArray,y::CuArray)
    maxPossibleThreads = 16
    Mthreads = min(M, maxPossibleThreads)
    Mblocks = ceil(Int, M/Mthreads)
    Nthreads = min(N, maxPossibleThreads)
    Nblocks = ceil(Int, N/Nthreads)
    CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks=(Mblocks, Nblocks) axpy_kernel_2d(alpha,x,y)
end

function dot_kernel(SIZE::Int, ret, x, y)
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

function reduce_kernel(SIZE::Int, red, ret)
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

function JACCTestCodes.dot(SIZE::Int, x::CuArray, y::CuArray)
    maxPossibleThreads = 512
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE/threads)
    ret = CUDA.zeros(Float64,blocks)
    rret = CUDA.zeros(Float64,1)
    CUDA.@sync @cuda threads=threads blocks=blocks shmem = 512 * sizeof(Float64) dot_kernel(SIZE, ret, x, y)
    CUDA.@sync @cuda threads=threads blocks=1 shmem = 512 * sizeof(Float64) reduce_kernel(blocks, ret, rret)
    return rret
end

#-------------------------2D DOT

function dot_kernel((M, N)::Tuple{Int, Int}, ret, x::CuArray, y::CuArray)
    shared_mem = @cuDynamicSharedMem(Float64, 16*16)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ti = threadIdx().x
    tj = threadIdx().y
    bi = blockIdx().x
    bj = blockIdx().y

    tmp::Float64 = 0.0
    shared_mem[((ti-1)*16)+tj] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds x[i,j] * y[i,j]
        shared_mem[(ti-1)*16+tj] = tmp
    end
    sync_threads()
    if (ti <= 8 && tj <= 8 && ti+8 <= M && tj+8 <= N)
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+7)*16)+(tj+8)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+8)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+7)*16)+tj]
    end
    sync_threads()
    if (ti <= 4 && tj <= 4 && ti+4 <= M && tj+4 <= N)
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+3)*16)+(tj+4)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+4)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+3)*16)+tj]
    end
    sync_threads()
    if (ti <= 2 && tj <= 2 && ti+2 <= M && tj+2 <= N)
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+1)*16)+(tj+2)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+2)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+1)*16)+tj]
    end
    sync_threads()
    if (ti == 1 && tj == 1 && ti+1 <= M && tj+1 <= N)
        shared_mem[((ti-1)*16)+tj] += shared_mem[ti*16+(tj+1)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+1)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[ti*16+tj]
        ret[bi,bj] = shared_mem[((ti-1)*16)+tj]
    end
    return nothing
end

function reduce_kernel((M, N)::Tuple{Int, Int}, red, ret)
    shared_mem = @cuDynamicSharedMem(Float64, 16*16)

    i = threadIdx().x
    j = threadIdx().y
    ii = i
    jj = j

    tmp::Float64 = 0.0
    shared_mem[(i-1)*16+j] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = threadIdx().y
            while jj <= N
                tmp = tmp + @inbounds red[ii,jj]
                jj += 16
            end
            ii += 16
        end
    elseif M > 16
        while ii <= N
            tmp = tmp + @inbounds red[ii,jj]
            ii += 16
        end
    elseif N > 16
        while jj <= N
            tmp = tmp + @inbounds red[ii,jj]
            jj += 16
        end
    elseif M <= 16 && N <= 16
        if i <= M && j <= N
            tmp = tmp + @inbounds red[i,j]
        end
    end
    shared_mem[(i-1)*16+j] = tmp
    red[i,j] = shared_mem[(i-1)*16+j]
    sync_threads()
    if (i <= 8 && j <= 8)
        if (i+8 <= M && j+8 <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[((i+7)*16)+(j+8)]
        end
        if (i <= M && j+8 <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+8)]
        end
        if (i+8 <= M && j <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[((i+7)*16)+j]
        end
    end
    sync_threads()
    if (i <= 4 && j <= 4)
        if (i+4 <= M && j+4 <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[((i+3)*16)+(j+4)]
        end
        if (i <= M && j+4 <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+4)]
        end
        if (i+4 <= M && j <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[((i+3)*16)+j]
        end
    end
    sync_threads()
    if (i <= 2 && j <= 2)
        if (i+2 <= M && j+2 <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[((i+1)*16)+(j+2)]
        end
        if (i <= M && j+2 <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+2)]
        end
        if (i+2 <= M && j <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[((i+1)*16)+j]
        end
    end
    sync_threads()
    if (i == 1 && j == 1)
        if (i+1 <= M && j+1 <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[i*16+(j+1)]
        end
        if (i <= M && j+1 <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+1)]
        end
        if (i+1 <= M && j <= N)
            shared_mem[((i-1)*16)+j] += shared_mem[i*16+j]
        end
        ret[1] = shared_mem[((i-1)*16)+j]
    end
    return nothing
end

function JACCTestCodes.dot((M,N)::Tuple{Int, Int}, x::CuArray, y::CuArray)
    maxPossibleThreads = 16
    Mthreads = min(M, maxPossibleThreads)
    Nthreads = min(N, maxPossibleThreads)
    Mblocks = ceil(Int, M/Mthreads)
    Nblocks = ceil(Int, N/Nthreads)
    ret = CUDA.zeros(Float64, (Mblocks, Nblocks))
    rret = CUDA.zeros(Float64,1)
    CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks = (Mblocks, Nblocks) shmem = 16 * 16 * sizeof(Float64) dot_kernel((M, N), ret, x, y)
    CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks = (1, 1) shmem = 16 * 16 * sizeof(Float64) reduce_kernel((Mblocks, Nblocks), ret, rret)
    return rret
end

#-------------------------LBM

function lbm_kernel(f, f1, f2, t, w, cx, cy, SIZE)

    x = ( blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = ( blockIdx().y - 1) * blockDim().y + threadIdx().y

    u = 0.0
    v = 0.0
    p = 0.0
    x_stream::Int32 = 0
    y_stream::Int32 = 0
    ind::Int32 = 0
    iind::Int32 = 0
    k::Int32 = 0

    if x > 1 && x < SIZE && y > 1 && y < SIZE
        for k in 1:9
            x_stream = x - cx[k]
            y_stream = y - cy[k]
            ind =  ( k - 1 ) * SIZE * SIZE + x * SIZE + y
            iind = ( k - 1 ) * SIZE * SIZE + x_stream * SIZE + y_stream
            f[trunc(Int,ind)] = f1[trunc(Int,iind)]
        end
        for k in 1:9
            ind =  ( k - 1 ) * SIZE * SIZE + x * SIZE + y
            p = p[1,1] + f[ind]
            u = u[1,1] + f[ind] * cx[k]
            v = v[1,1] + f[ind] * cy[k]
        end
        u = u / p
        v = v / p
        for k in 1:9
            feq = w[k] * p * ( 1.0 + 3.0 * ( cx[k] * u + cy[k] * v ) + ( cx[k] * u + cy[k] * v ) * ( cx[k] * u + cy[k] * v ) - 1.5 * ( ( u * u ) + ( v * v ) ) )
            ind =  ( k - 1 ) * SIZE * SIZE + x * SIZE + y
            iind = ( k - 1 ) * SIZE * SIZE + x_stream * SIZE + y_stream
            f2[trunc(Int,iind)] = f[trunc(Int,ind)] * (1.0 - 1.0 / t) + feq * 1 / t
        end
    end
    return nothing
end

function JACCTestCodes.lbm((M,N), f, f1, f2, t, w, cx::CuArray, cy::CuArray, SIZE)
    maxPossibleThreads = 16
    Mthreads = min(M, maxPossibleThreads)
    Mblocks = ceil(Int, M/Mthreads)
    Nthreads = min(N, maxPossibleThreads)
    Nblocks = ceil(Int, N/Nthreads)
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) lbm_kernel(f, f1, f2, t, w, cx, cy, SIZE)
end

function matvecmul_kernel(SIZE, a3, a2, a1, x, y)
    i = ( blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i == 1
        @inbounds y[i] = a2[i] * x[i] + a1[i] * x[i+1]
    elseif i == length(x)
        @inbounds y[i] = a3[i] * x[i-1] + a2[i] * x[i]
    else
        @inbounds y[i] = a3[i] * x[i-1] + a2[i] * x[i] + a1[i] * x[i+1]
    end
    return nothing
end

function JACCTestCodes.matvecmul(SIZE, a3, a2, a1, x::CuArray, y::CuArray)
    maxPossibleThreads = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE/threads)
    CUDA.@sync @cuda threads=threads blocks=blocks matvecmul_kernel(SIZE, a3, a2, a1, x, y)
end
end

