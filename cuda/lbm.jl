import Pkg
Pkg.activate(@__DIR__)


using CUDA
using JACC


function lbm_cuda_kernel(f, f1, f2, t, w, cx, cy, SIZE)

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

function lbm_cuda((M,N), f, f1, f2, t, w, cx, cy, SIZE)
  maxPossibleThreads = 16 
  Mthreads = min(M, maxPossibleThreads)
  Mblocks = ceil(Int, M/Mthreads)
  Nthreads = min(N, maxPossibleThreads)
  Nblocks = ceil(Int, N/Nthreads)
  CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) lbm_cuda_kernel(f, f1, f2, t, w, cx, cy, SIZE)
end

SIZE = 1000
f = ones(SIZE * SIZE * 9) .* 2.0
f1 = ones(SIZE * SIZE * 9) .* 3.0 
f2 = ones(SIZE * SIZE * 9) .* 4.0
cx = zeros(9)
cy = zeros(9)
cx[1] = 0
cy[1] = 0
cx[2] = 1
cy[2] = 0
cx[3] = -1
cy[3] = 0
cx[4] = 0
cy[4] = 1
cx[5] = 0
cy[5] = -1
cx[6] = 1
cy[6] = 1
cx[7] = -1
cy[7] = 1
cx[8] = -1
cy[8] = -1
cx[9] = 1
cy[9] = -1
 

println("CUDA:")

w   = ones(9)
t   = 1.0
df  = CuArray(f)
df1 = CuArray(f1)
df2 = CuArray(f2)
dcx = CuArray(cx)
dcy = CuArray(cy)
dw  = CuArray(w)

# warmup
lbm_cuda((100,100),df,df1,df2,t,dw,dcx,dcy,SIZE)

for i in [100,200,300,400,500,600,700,800,900,1000]
 @time begin
  lbm_cuda((i,i),df,df1,df2,t,dw,dcx,dcy,SIZE)
 end
end

function lbm(x, y, f, f1, f2, t, w, cx, cy, SIZE)
  u = 0.0
  v = 0.0
  p = 0.0
  x_stream = 0 
  y_stream = 0
  
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
end


println("JACC")

f = ones(SIZE * SIZE * 9) .* 2.0
f1 = ones(SIZE * SIZE * 9) .* 3.0 
f2 = ones(SIZE * SIZE * 9) .* 4.0
w   = ones(9)
t   = 1.0
jf  = JACC.Array(f)
jf1 = JACC.Array(f1)
jf2 = JACC.Array(f2)
jcx = JACC.Array(cx)
jcy = JACC.Array(cy)
jw  = JACC.Array(w)

# warmup
JACC.parallel_for((100,100),lbm,jf,jf1,jf2,t,jw,jcx,jcy,SIZE)

for i in [100,200,300,400,500,600,700,800,900,1000]
 @time begin
   JACC.parallel_for((i,i),lbm,jf,jf1,jf2,t,jw,jcx,jcy,SIZE)
 end
end
