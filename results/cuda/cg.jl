
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

function matvecmul_cuda_kernel(SIZE, a3, a2, a1, x, y)
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

function matvecmul_cuda(SIZE, a3, a2, a1, x, y)
  maxPossibleThreads = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X) 
  threads = min(SIZE, maxPossibleThreads)
  blocks = ceil(Int, SIZE/threads)
  CUDA.@sync @cuda threads=threads blocks=blocks matvecmul_cuda_kernel(SIZE, a3, a2, a1, x, y)
end

function cg_cuda(SIZE, a3, a2, a1, r, p, s, x, r_old, r_aux )

  a1 = a1 * 4.0
  r = r * 0.5
  p = p * 0.5
  alpha::Float64 = 0.0
  negative_alpha::Float64 = 0.0
  beta::Float64 = 0.0

  for i in 1:1

    r_old = copy(r) 

    matvecmul_cuda(SIZE, a3, a2, a1, p, s)

    # alpha0 = dot_amdgpu(SIZE, r, r)
    # alpha1 = dot_amdgpu(SIZE, p, s)
    alpha0 = dot_cuda(SIZE, r, r)
    alpha1 = dot_cuda(SIZE, p, s)

    # aalpha0::Float64 = alpha0[1::Integer]
    # aalpha1::Float64 = alpha1[1::Integer]
    CUDA.@allowscalar aalpha0::Float64 = alpha0[1::Integer]
    CUDA.@allowscalar aalpha1::Float64 = alpha1[1::Integer]

    alpha = aalpha0 / aalpha1
    negative_alpha = alpha * (-1.0)

    axpy_cuda(SIZE, negative_alpha, r, s)
    axpy_cuda(SIZE, alpha, x, p)

    beta0 = dot_cuda(SIZE, r, r)
    beta1 = dot_cuda(SIZE, r_old, r_old)
    
    # bbeta0::Float64 = beta0[1::Integer]
    # bbeta1::Float64 = beta1[1::Integer]
    CUDA.@allowscalar bbeta0::Float64 = beta0[1::Integer]
    CUDA.@allowscalar bbeta1::Float64 = beta1[1::Integer]

    beta = bbeta0 / bbeta1

    r_aux = copy(r)

    axpy_cuda(SIZE, beta, r_aux,p)
    cond = dot_cuda(SIZE, r, r)

    p = copy(r_aux)

  end
end


println("CUDA:")

SIZE = 100000000
a3 = ones(SIZE)
a2 = ones(SIZE)
a1 = ones(SIZE)
r = ones(SIZE)
p = ones(SIZE)
s = zeros(SIZE)
x = zeros(SIZE)
r_old = zeros(SIZE)
r_aux = zeros(SIZE)
da3 = CuArray(a3)
da2 = CuArray(a2)
da1 = CuArray(a1)
dr = CuArray(r)
dp = CuArray(p)
ds = CuArray(s)
dx = CuArray(x)
dr_old = CuArray(r_old)
dr_aux = CuArray(r_aux)

# warmup
cg_cuda(SIZE, da3, da2, da1, dr, dp, ds, dx, dr_old, dr_aux)

@time begin
  cg_cuda(SIZE, da3, da2, da1, dr, dp, ds, dx, dr_old, dr_aux)
end

function matvecmul(i, a3, a2, a1, x, y)
  if i == 1
    @inbounds y[i] = a2[i] * x[i] + a1[i] * x[i+1]
  elseif i == length(x)
    @inbounds y[i] = a3[i] * x[i-1] + a2[i] * x[i]
  else
    @inbounds y[i] = a3[i] * x[i-1] + a2[i] * x[i] + a1[i] * x[i+1]
  end
end

function axpy(i, alpha, x, y)
  @inbounds x[i] += alpha * y[i]
end

function dot(i, x, y)
  return  @inbounds x[i] * y[i]
end

function cg(SIZE, a3, a2, a1, r, p, s, x, r_old, r_aux )

  a1 = a1 * 4.0
  r = r * 0.5
  p = p * 0.5
  alpha::Float64 = 0.0
  negative_alpha::Float64 = 0.0
  beta::Float64 = 0.0

  for i in 1:1

    r_old = copy(r) 

    JACC.parallel_for(SIZE, matvecmul, a3, a2, a1, p, s)

    alpha0 = JACC.parallel_reduce(SIZE, dot, r, r)
    alpha1 = JACC.parallel_reduce(SIZE, dot, p, s)
    
    # aalpha0::Float64 = alpha0[1::Integer]
    # aalpha1::Float64 = alpha1[1::Integer]
    CUDA.@allowscalar aalpha0::Float64 = alpha0[1::Integer]
    CUDA.@allowscalar aalpha1::Float64 = alpha1[1::Integer]

    alpha = aalpha0 / aalpha1
    negative_alpha = alpha * (-1.0)

    JACC.parallel_for(SIZE, axpy, negative_alpha, r, s)
    JACC.parallel_for(SIZE, axpy, alpha, x, p)

    beta0 = JACC.parallel_reduce(SIZE, dot, r, r)
    beta1 = JACC.parallel_reduce(SIZE, dot, r_old, r_old)

    # bbeta0::Float64 = beta0[1::Integer]
    # bbeta1::Float64 = beta1[1::Integer]
    CUDA.@allowscalar bbeta0::Float64 = beta0[1::Integer]
    CUDA.@allowscalar bbeta1::Float64 = beta1[1::Integer]
    
    beta = bbeta0 / bbeta1

    r_aux = copy(r)

    JACC.parallel_for(SIZE, axpy, beta, r_aux, p)
    cond= JACC.parallel_reduce(SIZE, dot, r, r)

    p = copy(r_aux)

  end
end


println("JACC")

a3 = ones(SIZE)
a2 = ones(SIZE)
a1 = ones(SIZE)
r = ones(SIZE)
p = ones(SIZE)
s = zeros(SIZE)
x = zeros(SIZE)
r_old = zeros(SIZE)
r_aux = zeros(SIZE)
ja3 = JACC.Array(a3)
ja2 = JACC.Array(a2)
ja1 = JACC.Array(a1)
jr = JACC.Array(r)
jp = JACC.Array(p)
js = JACC.Array(s)
jx = JACC.Array(x)
jr_old = JACC.Array(r_old)
jr_aux = JACC.Array(r_aux)

# warmup
cg(SIZE, ja3, ja2, ja1, jr, jp, js, jx, jr_old, jr_aux)

@time begin
 cg(SIZE, ja3, ja2, ja1, jr, jp, js, jx, jr_old, jr_aux)
end
