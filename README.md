# JACC.jl Test Codes

## Run benchmarks (CUDA)
```bash
git clone git@github.com:pedrovalerolara/JACC-Test-Codes.git
cd JACC-Test-Codes/benchmarks/CUDA
julia --project=.
```

Instantiate environment
```julia
] dev ../.. JACC
```
Execute Benchmarks
```bash
julia --project=. benchmark.jl >& cuda.txt
```
