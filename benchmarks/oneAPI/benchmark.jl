using oneAPI
using JACCTestCodes
using KernelAbstractions
using JACCTestCodes.JACC

const KA = KernelAbstractions

run_benchmarks(OneBackend())
