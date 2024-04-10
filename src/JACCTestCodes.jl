module JACCTestCodes
using JACC
using KernelAbstractions
using Adapt
using BenchmarkTools
import BenchmarkTools: @btime

const KA = KernelAbstractions

export run_benchmarks

function run_benchmarks(backend::KA.Backend; only=nothing, only_api=nothing)
    SIZE = 1000
    alpha = 2.0
    if only == nothing || only == :axpy_1d
        benchmark_1d_axpy(backend, alpha, SIZE; only=only_api)
    end
    SIZE = 1000
    if only == nothing || only == :dot_1d
        benchmark_1d_dot(backend, SIZE; only=only_api)
    end
    SIZE = 300
    if only == nothing || only == :axpy_2d
        benchmark_2d_axpy(backend, alpha, SIZE; only=only_api)
    end
    SIZE = 300
    if only == nothing || only == :dot_2d
        benchmark_2d_dot(backend, SIZE; only=only_api)
    end
    SIZE = 100
    if only == nothing || only == :lbm
        benchmark_lbm(backend, SIZE; only=only_api)
    end
    if only == nothing || only == :cg
        benchmark_cg(backend, SIZE; only=only_api)
    end
end

function axpy(i, alpha, x, y) end
function axpy_jacc(i, alpha, x, y)
    @inbounds x[i] += alpha * y[i]
end

function dot(i, x, y) end
function dot_jacc(i, x, y)
    return @inbounds x[i] * y[i]
end

function axpy(i, j, alpha, x, y) end
function axpy_jacc(i, j, alpha, x, y)
    @inbounds x[i,j] = x[i,j] + alpha * y[i,j]
end

function dot(i, j, x, y) end

function dot_jacc(i, j, x, y)
    return @inbounds x[i,j] * y[i,j]
end

function lbm(x, y, f, f1, f2, t, w, cx, cy, SIZE) end
function lbm_jacc(x, y, f, f1, f2, t, w, cx, cy, SIZE)
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

function benchmark_1d_axpy(backend::KA.Backend, alpha, SIZE; only=nothing)
    x = ones(SIZE)
    y = ones(SIZE)
    if only != :jacc
        dx = adapt(backend, x)
        dy = adapt(backend, y)
        for i in [10,100]#,1000,10000]#,100000,1000000,10000000,100000000]
            println("axpy 1d $i")
            @btime begin
                axpy($i,$alpha,$dx,$dy)
            end
        end
    end
    if only != :vendor
        jx = JACC.Array(x)
        jy = JACC.Array(y)
        for i in [10,100]#,1000]#,10000]#,100000,1000000]#,10000000,100000000]
            println("axpy 1d jacc $i")
            @btime begin
                JACC.parallel_for($i, $axpy_jacc, $alpha, $jx, $jy)
            end
        end
    end
end

function benchmark_1d_dot(backend::KA.Backend, SIZE; only=nothing)
    x = ones(SIZE)
    y = ones(SIZE)
    if only != :jacc
        dx = adapt(backend, x)
        dy = adapt(backend, y)
        for i in [10,100]#,1000,10000]#,100000,1000000]#,10000000,100000000]
            println("dot 1d $i")
            @btime begin
                res = dot($i,$dx,$dy)
            end
        end
    end
    if only != :vendor
        jx = JACC.Array(x)
        jy = JACC.Array(y)
        for i in [10,100]#,1000,10000]#,100000,1000000]#,10000000,100000000]
            println("dot 1d jacc $i")
            @btime begin
                JACC.parallel_reduce($i, $dot_jacc, $jx, $jy)
            end
        end
    end
end

function benchmark_2d_axpy(backend::KA.Backend, alpha, SIZE; only=nothing)
    x = ones(SIZE,SIZE)
    y = ones(SIZE,SIZE)
    if only != :jacc
        dx = adapt(backend, x)
        dy = adapt(backend, y)
        for i in [100,200,300]#,4000,5000,6000,7000]#,8000,9000,10000]
            println("axpy 2d $i")
            @btime begin
                axpy(($i,$i),$alpha,$dx,$dy)
            end
        end
    end
    if only != :vendor
        jx = JACC.Array(x)
        jy = JACC.Array(y)
        for i in [100,200,300]#,4000,5000,6000,7000]#,8000,9000,10000]
            println("axpy jacc 2d $i")
            @btime begin
                JACC.parallel_for(($i,$i), $axpy_jacc, $alpha, $jx, $jy)
            end
        end
    end
end

function benchmark_2d_dot(backend::KA.Backend, SIZE; only=nothing)
    x = ones(SIZE,SIZE)
    y = ones(SIZE,SIZE)
    if only != :jacc
        dx = adapt(backend, x)
        dy = adapt(backend, y)
        for i in [100,200,300]#,4000,5000,6000,7000]#,8000,9000,10000]
            println("dot 2d $i")
            @btime begin
                res = dot(($i,$i),$dx,$dy)
            end
        end
    end
    if only != :vendor
        jx = JACC.Array(x)
        jy = JACC.Array(y)
        for i in [100,200,300]#,4000,5000,6000,7000]#,8000,9000,10000]
            println("dot jacc 2d $i")
            @btime begin
                res = JACC.parallel_reduce(($i,$i), $dot_jacc, $jx, $jy)
            end
        end
    end
end

function benchmark_lbm(backend::KA.Backend, SIZE; only=nothing)
    SIZE = 100
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

    w   = ones(9)
    t   = 1.0
    if only != :jacc
        df  = adapt(backend, f)
        df1 = adapt(backend, f1)
        df2 = adapt(backend, f2)
        dcx = adapt(backend, cx)
        dcy = adapt(backend, cy)
        dw  = adapt(backend, w)
        for i in [100,200,300]#,400,500,600,700]#,800,900,1000]
            println("lbm $i")
            @btime begin
                lbm(($i,$i),$df,$df1,$df2,$t,$dw,$dcx,$dcy,$SIZE)
            end
        end
    end
    if only != :vendor
        jf  = JACC.Array(f)
        jf1 = JACC.Array(f1)
        jf2 = JACC.Array(f2)
        jcx = JACC.Array(cx)
        jcy = JACC.Array(cy)
        jw  = JACC.Array(w)
        for i in [100,200,300]#,400,500,600,700]#,800,900,1000]
            println("lbm jacc $i")
            @btime begin
                JACC.parallel_for(($i,$i),$lbm_jacc,$jf,$jf1,$jf2,$t,$jw,$jcx,$jcy,$SIZE)
            end
        end
    end
end

function benchmark_cg(backend::KA.Backend, SIZE; only=nothing)
    SIZE = 1000
    a3 = ones(SIZE)
    a2 = ones(SIZE)
    a1 = ones(SIZE)
    r = ones(SIZE)
    p = ones(SIZE)
    s = zeros(SIZE)
    x = zeros(SIZE)
    r_old = zeros(SIZE)
    r_aux = zeros(SIZE)
    if only != :jacc
        da3 = adapt(backend, a3)
        da2 = adapt(backend, a2)
        da1 = adapt(backend, a1)
        dr = adapt(backend, r)
        dp = adapt(backend, p)
        ds = adapt(backend, s)
        dx = adapt(backend, x)
        dr_old = adapt(backend, r_old)
        dr_aux = adapt(backend, r_aux)

        println("cg")
        @btime begin
            cg($SIZE, $da3, $da2, $da1, $dr, $dp, $ds, $dx, $dr_old, $dr_aux)
        end

    end
    if only != :vendor
        ja3 = JACC.Array(a3)
        ja2 = JACC.Array(a2)
        ja1 = JACC.Array(a1)
        jr = JACC.Array(r)
        jp = JACC.Array(p)
        js = JACC.Array(s)
        jx = JACC.Array(x)
        jr_old = JACC.Array(r_old)
        jr_aux = JACC.Array(r_aux)

        println("cg jacc")
        @btime begin
            cg_jacc($SIZE, $ja3, $ja2, $ja1, $jr, $jp, $js, $jx, $jr_old, $jr_aux)
        end
    end
end

function matvecmul(i, a3, a2, a1, x, y) end
function matvecmul_jacc(i, a3, a2, a1, x, y)
  if i == 1
    @inbounds y[i] = a2[i] * x[i] + a1[i] * x[i+1]
  elseif i == length(x)
    @inbounds y[i] = a3[i] * x[i-1] + a2[i] * x[i]
  else
    @inbounds y[i] = a3[i] * x[i-1] + a2[i] * x[i] + a1[i] * x[i+1]
  end
end

function cg_jacc(SIZE, a3, a2, a1, r, p, s, x, r_old, r_aux)
    a1 = a1 * 4.0
    r = r * 0.5
    p = p * 0.5
    alpha::Float64 = 0.0
    negative_alpha::Float64 = 0.0
    beta::Float64 = 0.0

    for i in 1:1

        r_old = copy(r)

        JACC.parallel_for(SIZE, matvecmul_jacc, a3, a2, a1, p, s)

        alpha0 = JACC.parallel_reduce(SIZE, dot_jacc, r, r)
        alpha1 = JACC.parallel_reduce(SIZE, dot_jacc, p, s)

        aalpha0::Float64 = alpha0[1::Integer]
        aalpha1::Float64 = alpha1[1::Integer]

        alpha = aalpha0 / aalpha1
        negative_alpha = alpha * (-1.0)

        JACC.parallel_for(SIZE, axpy_jacc, negative_alpha, r, s)
        JACC.parallel_for(SIZE, axpy_jacc, alpha, x, p)

        beta0 = JACC.parallel_reduce(SIZE, dot_jacc, r, r)
        beta1 = JACC.parallel_reduce(SIZE, dot_jacc, r_old, r_old)
        bbeta0::Float64 = beta0[1::Integer]
        bbeta1::Float64 = beta1[1::Integer]

        beta = bbeta0 / bbeta1

        r_aux = copy(r)

        JACC.parallel_for(SIZE, axpy_jacc, beta, r_aux, p)
        cond= JACC.parallel_reduce(SIZE, dot_jacc, r, r)

        p = copy(r_aux)

    end
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

    matvecmul(SIZE, a3, a2, a1, p, s)

    alpha0 = dot(SIZE, r, r)
    alpha1 = dot(SIZE, p, s)

    aalpha0::Float64 = alpha0[1::Integer]
    aalpha1::Float64 = alpha1[1::Integer]

    alpha = aalpha0 / aalpha1
    negative_alpha = alpha * (-1.0)

    axpy(SIZE, negative_alpha, r, s)
    axpy(SIZE, alpha, x, p)

    beta0 = dot(SIZE, r, r)
    beta1 = dot(SIZE, r_old, r_old)

    bbeta0::Float64 = beta0[1::Integer]
    bbeta1::Float64 = beta1[1::Integer]

    beta = bbeta0 / bbeta1

    r_aux = copy(r)

    axpy(SIZE, beta, r_aux,p)
    cond = dot(SIZE, r, r)

    p = copy(r_aux)

  end
end
end
