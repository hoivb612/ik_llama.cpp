git clone https://github.com/hoivb612/ik_llama.cpp .
Cloning into '.'...
remote: Enumerating objects: 20672, done.
remote: Counting objects: 100% (1292/1292), done.
remote: Compressing objects: 100% (625/625), done.
remote: Total 20672 (delta 696), reused 1147 (delta 638), pack-reused 19380 (from 1)
Receiving objects: 100% (20672/20672), 32.96 MiB | 25.55 MiB/s, done.
Resolving deltas: 100% (14879/14879), done.


==============================================

09/24/2024

C:\llama.cpp\llama.ik_master>git pull
remote: Enumerating objects: 192, done.
remote: Counting objects: 100% (192/192), done.
remote: Compressing objects: 100% (107/107), done.
remote: Total 192 (delta 122), reused 132 (delta 84), pack-reused 0 (from 0)
Receiving objects: 100% (192/192), 759.18 KiB | 16.87 MiB/s, done.
Resolving deltas: 100% (122/122), completed with 6 local objects.
From https://github.com/hoivb612/ik_llama.cpp
   9b53c253..be579129  main       -> origin/main
warning: fetch normally indicates which branches had a forced update,
but that check has been disabled; to re-enable, use '--show-forced-updates'
flag or run 'git config fetch.showForcedUpdates true'
Updating 9b53c253..be579129
Fast-forward
 common/common.cpp                    |    3 +
 examples/llama-bench/llama-bench.cpp |    3 +
 examples/quantize/quantize.cpp       |    1 +
 ggml/include/ggml.h                  |   51 +-
 ggml/src/CMakeLists.txt              |    4 +
 ggml/src/ggml-common.h               |   62 +-
 ggml/src/ggml-cuda.cu                |   54 +-
 ggml/src/ggml-cuda/common.cuh        |    7 +
 ggml/src/ggml-cuda/convert.cu        |  193 ++-
 ggml/src/ggml-cuda/convert.cuh       |    5 +-
 ggml/src/ggml-cuda/fattn-common.cuh  |    4 +-
 ggml/src/ggml-cuda/iqk_mmvq.cu       |   87 +-
 ggml/src/ggml-cuda/iqk_mmvq.cuh      |    4 +
 ggml/src/ggml-cuda/mmvq.cu           |    3 +
 ggml/src/ggml-cuda/norm.cu           |   75 +
 ggml/src/ggml-cuda/norm.cuh          |    2 +
 ggml/src/ggml-cuda/vendors/cuda.h    |    1 +
 ggml/src/ggml-cuda/vendors/hip.h     |    1 +
 ggml/src/ggml-metal.m                |  101 +-
 ggml/src/ggml-metal.metal            |  543 ++++++-
 ggml/src/ggml-quants.c               |    1 +
 ggml/src/ggml.c                      |  191 ++-
 ggml/src/iqk/iqk_mul_mat.cpp         | 2897 ++++++++++++++++++++++++++++------
 ggml/src/iqk/iqk_mul_mat.h           |    4 +-
 ggml/src/iqk/iqk_quantize.cpp        |   89 +-
 ggml/src/iqk/iqk_quantize.h          |    6 +
 include/llama.h                      |   18 +-
 src/llama.cpp                        |   54 +-
 28 files changed, 3720 insertions(+), 744 deletions(-)


