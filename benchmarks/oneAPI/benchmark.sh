
pwd
echo $JULIA_DEPOT_PATH
echo
# julia --project=. benchmark.jl jacc axpy_1d    >& ../../results/oneAPI/jacc_axpy_1d.txt
julia --project=. benchmark.jl jacc axpy_2d    >& ../../results/oneAPI/jacc_axpy_2d.txt
# julia --project=. benchmark.jl jacc dot_1d     >& ../../results/oneAPI/jacc_dot_1d.txt
# julia --project=. benchmark.jl jacc dot_2d     >& ../../results/oneAPI/jacc_dot_2d.txt
# julia --project=. benchmark.jl jacc lbm        >& ../../results/oneAPI/jacc_lbm.txt
# julia --project=. benchmark.jl jacc cg         >& ../../results/oneAPI/jacc_cg.txt
# julia --project=. benchmark.jl vendor axpy_1d  >& ../../results/oneAPI/vendor_axpy_1d.txt
# julia --project=. benchmark.jl vendor axpy_2d  >& ../../results/oneAPI/vendor_axpy_2d.txt
# julia --project=. benchmark.jl vendor dot_1d   >& ../../results/oneAPI/vendor_dot_1d.txt
# julia --project=. benchmark.jl vendor dot_2d   >& ../../results/oneAPI/vendor_dot_2d.txt
# julia --project=. benchmark.jl vendor lbm      >& ../../results/oneAPI/vendor_lbm.txt
# julia --project=. benchmark.jl vendor cg       >& ../../results/oneAPI/vendor_cg.txt