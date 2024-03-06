using CUDA
using JACCTestCodes
using KernelAbstractions
using JACCTestCodes.JACC

const KA = KernelAbstractions

CUDA.allowscalar() do
run_benchmarks(CUDABackend())
end
