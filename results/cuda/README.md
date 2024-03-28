# CUDA Runs on Perlmutter and Polaris


## NERSC Perlmutter

To run:
```
$ ml load julia/1.9.4
$ julia <script name>
```

Each script activates `Project.toml` and `Manifest.toml` in this directory:

```julia
import Pkg
Pkg.activate(@__DIR__)
```

There are two kinds of GPU nodes available -- which have two different generations of Nvidia A100 GPUs: https://docs.nersc.gov/systems/perlmutter/architecture/

We iused the local CUDA runtime installation. More info on earch below:

### 40GB nodes:

```
julia> CUDA.versioninfo()
CUDA runtime 12.2, local installation
CUDA driver 12.3
NVIDIA driver 525.105.17, originally for CUDA 12.2

CUDA libraries:
- CUBLAS: 12.2.1
- CURAND: 10.3.3
- CUFFT: 11.0.8
- CUSOLVER: 11.5.0
- CUSPARSE: 12.1.1
- CUPTI: 20.0.0
- NVML: 12.0.0+525.105.17

Julia packages:
- CUDA: 5.2.0
- CUDA_Driver_jll: 0.7.0+1
- CUDA_Runtime_jll: 0.11.1+0
- CUDA_Runtime_Discovery: 0.2.3

Toolchain:
- Julia: 1.9.4
- LLVM: 14.0.6

Preferences:
- CUDA_Runtime_jll.version: 12.2
- CUDA_Runtime_jll.local: true

4 devices:
  0: NVIDIA A100-SXM4-40GB (sm_80, 39.389 GiB / 40.000 GiB available)
  1: NVIDIA A100-SXM4-40GB (sm_80, 39.389 GiB / 40.000 GiB available)
  2: NVIDIA A100-SXM4-40GB (sm_80, 39.389 GiB / 40.000 GiB available)
  3: NVIDIA A100-SXM4-40GB (sm_80, 39.389 GiB / 40.000 GiB available)
```

### 80GB nodes:

```
julia> CUDA.versioninfo()
CUDA runtime 12.2, local installation
CUDA driver 12.3
NVIDIA driver 525.105.17, originally for CUDA 12.2

CUDA libraries:
- CUBLAS: 12.2.1
- CURAND: 10.3.3
- CUFFT: 11.0.8
- CUSOLVER: 11.5.0
- CUSPARSE: 12.1.1
- CUPTI: 20.0.0
- NVML: 12.0.0+525.105.17

Julia packages:
- CUDA: 5.2.0
- CUDA_Driver_jll: 0.7.0+1
- CUDA_Runtime_jll: 0.11.1+0
- CUDA_Runtime_Discovery: 0.2.3

Toolchain:
- Julia: 1.9.4
- LLVM: 14.0.6

Preferences:
- CUDA_Runtime_jll.version: 12.2
- CUDA_Runtime_jll.local: true

4 devices:
  0: NVIDIA A100-SXM4-80GB (sm_80, 79.150 GiB / 80.000 GiB available)
  1: NVIDIA A100-SXM4-80GB (sm_80, 79.150 GiB / 80.000 GiB available)
  2: NVIDIA A100-SXM4-80GB (sm_80, 79.150 GiB / 80.000 GiB available)
  3: NVIDIA A100-SXM4-80GB (sm_80, 79.150 GiB / 80.000 GiB available)
```
