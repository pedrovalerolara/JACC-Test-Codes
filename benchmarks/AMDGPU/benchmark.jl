using AMDGPU
using JACCTestCodes
using KernelAbstractions
using JACCTestCodes.JACC

const KA = KernelAbstractions

#AMDGPU.allowscalar() do
  run_benchmarks(ROCBackend())
#end
alpha=2.0
