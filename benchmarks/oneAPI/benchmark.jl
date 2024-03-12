using oneAPI
using JACCTestCodes
using KernelAbstractions
using JACCTestCodes.JACC

const KA = KernelAbstractions
backend = oneAPIBackend()
only = nothing
only_api = nothing
if length(ARGS) >= 2
    if ARGS[1] == "jacc"
        only_api = :jacc
    elseif ARGS[1] == "vendor"
        only_api = :vendor
    else
        error("Wrong argument $(ARGS[1])")
    end
    if ARGS[2] == "axpy_1d"
        only = :axpy_1d
    elseif ARGS[2] == "axpy_2d"
        only = :axpy_2d
    elseif ARGS[2] == "dot_1d"
        only = :dot_1d
    elseif ARGS[2] == "dot_2d"
        only = :dot_2d
    elseif ARGS[2] == "lbm"
        only = :lbm
    elseif ARGS[2] == "cg"
        only = :cg
    else
        error("Wrong argument $(ARGS[2])")
    end
end

oneAPI.allowscalar() do
    run_benchmarks(backend, only=only, only_api=only_api)
end
# run_benchmarks(oneAPIBackend(); only=:axpy_1d, only_api=:jacc)
SIZE = 100000000
alpha = 2.0
