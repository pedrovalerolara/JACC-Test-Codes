import Pkg
Pkg.activate(@__DIR__)


using CUDA
using JACC


function dot_cuda_kernel((M, N), ret, x, y)
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

function reduce_kernel((M, N), red, ret)
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

function dot_cuda((M,N), x, y)
  maxPossibleThreads = 16 
  Mthreads = min(M, maxPossibleThreads)
  Nthreads = min(N, maxPossibleThreads)
  Mblocks = ceil(Int, M/Mthreads)
  Nblocks = ceil(Int, N/Nthreads)
  ret = CUDA.zeros(Float64, (Mblocks, Nblocks))
  rret = CUDA.zeros(Float64,1)
  CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks = (Mblocks, Nblocks) shmem = 16 * 16 * sizeof(Float64) dot_cuda_kernel((M, N), ret, x, y)
  CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks = (1, 1) shmem = 16 * 16 * sizeof(Float64) reduce_kernel((Mblocks, Nblocks), ret, rret)
  return rret
end


println("CUDA:")

SIZE = 10000
x = ones(SIZE,SIZE)
y = ones(SIZE,SIZE)
dx = CuArray(x)
dy = CuArray(y)

# warmuo
dot_cuda((1000,1000),dx,dy)

for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
 @time begin
  res = dot_cuda((i,i),dx,dy)
 end
end

function dot(i, j, x, y)
  return @inbounds x[i,j] * y[i,j]
end


println("JACC")

x = ones(SIZE,SIZE)
y = ones(SIZE,SIZE)
jx = JACC.Array(x)
jy = JACC.Array(y)

# warmup
JACC.parallel_reduce((1000,1000), dot, jx, jy)

for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
 @time begin
  res = JACC.parallel_reduce((i,i), dot, jx, jy)
 end
end
