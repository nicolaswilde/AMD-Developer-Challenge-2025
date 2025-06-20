# This script provides a template for using load_inline to run a HIP kernel for
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
CPP_WRAPPER = """
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr const int BLOCK = 128;

// swizzle function for 8-fp8 / 128-fp8
#define SWZ(m, n) (((m) * 8 + (n)) % 128)

__launch_bounds__(512, 2)
__global__ void custom_kernel(const __hip_fp8_e4m3_fnuz* __restrict__ a,
                              const __hip_fp8_e4m3_fnuz* __restrict__ b,
                              const float* __restrict__ as,
                              const float* __restrict__ bs,
                              __hip_bfloat16* __restrict__ c, int m, int n, int k) {

    using float4 = float __attribute__((ext_vector_type(4)));

    __shared__ __hip_fp8_e4m3_fnuz lds_a[128][128];
    __shared__ __hip_fp8_e4m3_fnuz lds_b[128][128];

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int wave_m = (threadIdx.y / 2) * 32;
    int wave_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 16) * 4;
    int thread_c_n =  threadIdx.x % 16;

    // LDGSTS A&B Parameters
    int sts_wave_k  = (threadIdx.y / 4) * 32;
    int sts_wave_mn = (threadIdx.y % 4) * 32;
    int sts_thread_mn = (threadIdx.x / 8) * 4;
    int sts_thread_k  = (threadIdx.x % 8) * 4;

    int sts_k  = sts_wave_k + sts_thread_k;
    int sts_mn = sts_wave_mn + sts_thread_mn;

    int ldg_m = thread_block_m + sts_wave_mn + sts_thread_k;
    int ldg_n = thread_block_n + sts_wave_mn + sts_thread_k;
    int ldg_k = sts_wave_k + sts_thread_mn;

    // LDS A&B Parameters
    int lds_m = wave_m + threadIdx.x % 16;
    int lds_n = wave_n + threadIdx.x % 16;
    int lds_k = (threadIdx.x / 16) * 8;

    // C&D Parameters
    int c_m = thread_block_m + wave_m + thread_c_m;
    int c_n = thread_block_n + wave_n + thread_c_n;

    int as_m = c_m;
    int bs_n = c_n / BLOCK;

    float4 tile_d[2][4] = {0};

    for (int ib = 0; ib < k/BLOCK; ib++) {
        __hip_fp8_e4m3_fnuz ldg_a  [2][4][4], ldg_b  [2][4][4];
        __hip_fp8_e4m3_fnuz ldg_a_t[2][4][4], ldg_b_t[2][4][4];
        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 4; jj++) {
                *(float *)ldg_a[ii][jj] = ldg_m < m ? *(float *)&a[(ldg_k + ii*64 + jj + ib*BLOCK) * m + ldg_m] : 0;
                *(float *)ldg_b[ii][jj] = ldg_n < n ? *(float *)&b[(ldg_k + ii*64 + jj + ib*BLOCK) * n + ldg_n] : 0;
            }
        }

        float this_as[2][4];
        for (int ii = 0; ii < 2; ii++) {
            *(float4 *)this_as[ii] = as_m + ii*16 < m ? *(float4 *)&as[ib*m + as_m + ii*16] : 0;
        }
        float this_bs = bs[ib * ((n+BLOCK-1)/BLOCK) + bs_n];

        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 4; jj++) {
                for (int kk = 0; kk < 4; kk++) {
                    ldg_a_t[ii][jj][kk] = ldg_a[ii][kk][jj];
                    ldg_b_t[ii][jj][kk] = ldg_b[ii][kk][jj];
                }
            }
            float value_a[4], value_b[4];
            *(float4 *)value_a = *(float4 *)ldg_a_t[ii];
            *(float4 *)value_b = *(float4 *)ldg_b_t[ii];
            for (int jj = 0; jj < 4; jj++) {
                value_a[jj] = __shfl(value_a[jj], (threadIdx.x % 8) * 8 + (threadIdx.x / 8), 64);
                value_b[jj] = __shfl(value_b[jj], (threadIdx.x % 8) * 8 + (threadIdx.x / 8), 64);
                *(float *)&lds_a[sts_mn + jj][SWZ(sts_mn + jj, sts_k + ii*64)] = value_a[jj];
                *(float *)&lds_b[sts_mn + jj][SWZ(sts_mn + jj, sts_k + ii*64)] = value_b[jj];
            }
        }

        __syncthreads();

        float4 block_tile_d[2][4] = {0};
        long tile_a[2][4], tile_b[4][4];
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_a[ii][kk] = *(long *)&lds_a[lds_m + ii*16][SWZ(lds_m, lds_k + kk*32)];
            }
        }
        for (int jj = 0; jj < 4; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_b[jj][kk] = *(long *)&lds_b[lds_n + jj*16][SWZ(lds_n, lds_k + kk*32)];
            }
        }
        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 4; jj++) {
                for (int kk = 0; kk < 4; kk++) {
                    block_tile_d[ii][jj] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(tile_a[ii][kk], tile_b[jj][kk], block_tile_d[ii][jj], 0, 0, 0);
                }
                for (int kk = 0; kk < 4; kk++) {
                    tile_d[ii][jj][kk] += block_tile_d[ii][jj][kk] * this_as[ii][kk] * this_bs;
                }
            }
        }

        __syncthreads();
    }

    // finally, write the result as bf16
    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 4; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                if (c_m + ii*16 < m && c_n + jj*16 < n) {
                    c[(c_m + ii*16 + kk) * n + c_n + jj*16] = (__hip_bfloat16)(tile_d[ii][jj][kk]);
                }
            }
        }
    }
}

void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    custom_kernel<<<dim3((m+127)/128, (n+127)/128), dim3(64, 8), 0, 0>>> ((__hip_fp8_e4m3_fnuz*)a.data_ptr(), (__hip_fp8_e4m3_fnuz*)b.data_ptr(),
    as.data_ptr<float>(), bs.data_ptr<float>(), (__hip_bfloat16*)c.data_ptr(), m, n, k);
    //C10_CUDA_CHECK(cudaGetLastError());
}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='fp8_mm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['fp8_mm'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O3"],
)

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    module.fp8_mm(a, b, a_scale, b_scale, c)
    return c