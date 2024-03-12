module JACCTestCodesOneAPI
using oneAPI
using JACCTestCodes
import JACCTestCodes: dot, axpy, matvecmul, lbm

# #-------------------------1D AXPY

function axpy_kernel_1d(alpha,x,y)
    i = get_global_id()
    @inbounds x[i] = x[i] + alpha * y[i]
    return nothing
end
#-------------------------1D DOT

function axpy_kernel_2d(alpha,x,y)
    i = get_global_id(0)
    j = get_global_id(1)
    @inbounds x[i,j] = x[i,j] + alpha * y[i,j]
    return nothing
end

function JACCTestCodes.axpy(SIZE::Int,alpha,x::oneArray,y::oneArray)
    maxPossibleItems = 512
    items = min(SIZE, maxPossibleItems)
    groups = ceil(Int, SIZE / items)
    oneAPI.@sync @oneapi items = items groups = groups axpy_kernel_1d(alpha,x,y)
end

function JACCTestCodes.axpy((M,N)::Tuple{Int, Int},alpha,x::oneArray,y::oneArray)
    maxPossibleItems = 16
    Mitems = min(M, maxPossibleItems)
    Nitems = min(N, maxPossibleItems)
    Mgroups = ceil(Int, M / Mitems)
    Ngroups = ceil(Int, N / Nitems)
    oneAPI.@sync @oneapi items = (Mitems, Nitems) groups = (Mgroups, Ngroups) axpy_kernel_2d(alpha,x,y)
end

function dot_kernel(SIZE::Int, ret, x, y)
    shared_mem = oneLocalArray(Float64, 512)
    i = get_global_id(0)
    ti = get_local_id(0)
    tmp::Float64 = 0.0
    shared_mem[ti] = 0.0

    if i <= SIZE
        tmp = @inbounds x[i] * y[i]
        shared_mem[ti] = tmp
    end
    barrier()
    if (ti <= 256)
        shared_mem[ti] += shared_mem[ti+256]
    end
    barrier()
    if (ti <= 128)
        shared_mem[ti] += shared_mem[ti+128]
    end
    barrier()
    if (ti <= 64)
        shared_mem[ti] += shared_mem[ti+64]
    end
    barrier()
    if (ti <= 32)
        shared_mem[ti] += shared_mem[ti+32]
    end
    barrier()
    if (ti <= 16)
        shared_mem[ti] += shared_mem[ti+16]
    end
    barrier()
    if (ti <= 8)
        shared_mem[ti] += shared_mem[ti+8]
    end
    barrier()
    if (ti <= 4)
        shared_mem[ti] += shared_mem[ti+4]
    end
    barrier()
    if (ti <= 2)
        shared_mem[ti] += shared_mem[ti+2]
    end
    barrier()
    if (ti == 1)
        shared_mem[ti] += shared_mem[ti+1]
        ret[get_group_id(0)] = shared_mem[ti]
    end
    return nothing
end

function reduce_kernel(SIZE::Int, red, ret)
    shared_mem = oneLocalArray(Float64, 512)
    i = get_global_id(0)
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
    barrier()
    if (i <= 256)
        shared_mem[i] += shared_mem[i+256]
    end
    barrier()
    if (i <= 128)
        shared_mem[i] += shared_mem[i+128]
    end
    barrier()
    if (i <= 64)
        shared_mem[i] += shared_mem[i+64]
    end
    barrier()
    if (i <= 32)
        shared_mem[i] += shared_mem[i+32]
    end
    barrier()
    if (i <= 16)
        shared_mem[i] += shared_mem[i+16]
    end
    barrier()
    if (i <= 8)
        shared_mem[i] += shared_mem[i+8]
    end
    barrier()
    if (i <= 4)
        shared_mem[i] += shared_mem[i+4]
    end
    barrier()
    if (i <= 2)
        shared_mem[i] += shared_mem[i+2]
    end
    barrier()
    if (i == 1)
        shared_mem[i] += shared_mem[i+1]
        ret[1] = shared_mem[1]
    end
    return nothing
end

function JACCTestCodes.dot(SIZE::Int, x::oneArray, y::oneArray)
    numItems = 512
    items = min(SIZE, numItems)
    groups = ceil(Int, SIZE/items)
    ret = oneAPI.zeros(Float64, groups)
    rret = oneAPI.zeros(Float64, 1)
    oneAPI.@sync @oneapi items = items groups = groups dot_kernel(SIZE, ret, x, y)
    oneAPI.@sync @oneapi items = items groups = 1 reduce_kernel(SIZE, ret, rret)
    return rret
end

#-------------------------2D DOT

function dot_kernel((M, N)::Tuple{Int, Int}, ret, x, y)
    shared_mem = oneLocalArray(Float64, 16 * 16)
    i = get_global_id(0)
    j = get_global_id(1)
    ti = get_local_id(0)
    tj = get_local_id(1)
    bi = get_group_id(0)
    bj = get_group_id(1)

    tmp::Float64 = 0.0
    shared_mem[((ti-1)*16)+tj] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds x[i,j] * y[i,j]
        shared_mem[(ti-1)*16+tj] = tmp
    end
    barrier()
    if (ti <= 8 && tj <= 8 && ti+8 <= M && tj+8 <= N)
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+7)*16)+(tj+8)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+8)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+7)*16)+tj]
    end
    barrier()
    if (ti <= 4 && tj <= 4 && ti+4 <= M && tj+4 <= N)
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+3)*16)+(tj+4)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+4)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+3)*16)+tj]
    end
    barrier()
    if (ti <= 2 && tj <= 2 && ti+2 <= M && tj+2 <= N)
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+1)*16)+(tj+2)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+2)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+1)*16)+tj]
    end
    barrier()
    if (ti == 1 && tj == 1 && ti+1 <= M && tj+1 <= N)
        shared_mem[((ti-1)*16)+tj] += shared_mem[ti*16+(tj+1)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+1)]
        shared_mem[((ti-1)*16)+tj] += shared_mem[ti*16+tj]
        ret[bi,bj] = shared_mem[((ti-1)*16)+tj]
    end
    return nothing
end

function reduce_kernel((M, N)::Tuple{Int, Int}, red, ret)
    shared_mem = oneLocalArray(Float64, 16 * 16)
    i = get_local_id(0)
    j = get_local_id(1)
    ii = i
    jj = j

    tmp::Float64 = 0.0
    shared_mem[(i-1)*16+j] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = get_local_id(1)
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
    barrier()
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
    barrier()
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
    barrier()
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
    barrier()
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

function JACCTestCodes.dot((M,N)::Tuple{Int, Int}, x::oneArray, y::oneArray)
    maxPossibleItems = 16
    Mitems = min(M, maxPossibleItems)
    Nitems = min(N, maxPossibleItems)
    Mgroups = ceil(Int, M/Mitems)
    Ngroups = ceil(Int, N/Nitems)
    ret = oneAPI.zeros(Float64,(Mgroups, Ngroups))
    rret = oneAPI.zeros(Float64,1)
    oneAPI.@sync @oneapi items = (Mitems, Nitems) groups = (Mgroups, Ngroups) dot_kernel((M, N), ret, x, y)
    oneAPI.@sync @oneapi items = (Mitems, Nitems) groups = (1, 1) reduce_kernel((Mgroups, Ngroups), ret, rret)
    return rret
end

#-------------------------LBM

function lbm_kernel(f, f1, f2, t, w, cx, cy, SIZE)

    x = get_global_id(0)
    y = get_global_id(1)

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

function JACCTestCodes.lbm((M,N), f, f1, f2, t, w, cx::oneArray, cy::oneArray, SIZE)
    maxPossibleThreads = 16
    Mitems = min(M, maxPossibleThreads)
    Nitems = min(N, maxPossibleThreads)
    Mgroups = ceil(Int, M/Mitems)
    Ngroups = ceil(Int, N/Nitems)
    oneAPI.@sync @oneapi items = (Mitems, Nitems) groups = (Mgroups, Ngroups) lbm_kernel(f, f1, f2, t, w, cx, cy, SIZE)
end

function matvecmul_kernel(SIZE, a3, a2, a1, x, y)
    i = get_global_id()
    if i == 1
      @inbounds y[i] = a2[i] * x[i] + a1[i] * x[i+1]
    elseif i == length(x)
      @inbounds y[i] = a3[i] * x[i-1] + a2[i] * x[i]
    else
      @inbounds y[i] = a3[i] * x[i-1] + a2[i] * x[i] + a1[i] * x[i+1]
    end
    return nothing
end

function JACCTestCodes.matvecmul(SIZE, a3, a2, a1, x::oneArray, y::oneArray)
    maxPossibleItems = 512
    items = min(SIZE, maxPossibleItems)
    groups = ceil(Int, SIZE / items)
    oneAPI.@sync @oneapi items = items groups = groups matvecmul_kernel(SIZE, a3, a2, a1, x, y)
end

end

