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

__launch_bounds__(256, 2)
__global__ void custom_kernel(const __hip_fp8_e4m3_fnuz* __restrict__ a,
                              const __hip_fp8_e4m3_fnuz* __restrict__ b,
                              const float* __restrict__ as,
                              const float* __restrict__ bs,
                              __hip_bfloat16* __restrict__ c, int m, int n, int k) {

    using float4 = float __attribute__((ext_vector_type(4)));

    __shared__ __hip_fp8_e4m3_fnuz lds_a[2][64][128];
    __shared__ __hip_fp8_e4m3_fnuz lds_b[2][64][128];

    int thread_block_m = blockIdx.x * 64;
    int thread_block_n = blockIdx.y * 64;

    int wave_m = (threadIdx.y / 2) * 32;
    int wave_n = (threadIdx.y % 2) * 32;

    int thread_c_m = (threadIdx.x / 16) * 4;
    int thread_c_n =  threadIdx.x % 16;

    // LDGSTS A&B Parameters
    int sts_mn = threadIdx.y * 8 + threadIdx.x / 8;
    int sts_k  = (threadIdx.x % 8) * 16;
    int ldg_m = thread_block_m + sts_mn;
    int ldg_n = thread_block_n + sts_mn;
    int ldg_k = sts_k;

    // LDS A&B Parameters
    int lds_m = wave_m + threadIdx.x % 16;
    int lds_n = wave_n + threadIdx.x % 16;
    int lds_k = (threadIdx.x / 16) * 8;

    // C&D Parameters
    int c_m = thread_block_m + wave_m + thread_c_m;
    int c_n = thread_block_n + wave_n + thread_c_n;

    int as_m = c_m;
    int bs_n = c_n / BLOCK;

    float4 tile_d[2][2] = {0};

    { // fiist LDGSTS
        __hip_fp8_e4m3_fnuz ldg_a[2][16], ldg_b[2][16];
        for (int ii = 0; ii < 2; ii++) {
            *(float4 *)ldg_a[ii] = ldg_m + ii*32 < m ? *(float4 *)&a[(ldg_m + ii*32) * k + ldg_k] : 0;
            *(float4 *)ldg_b[ii] = ldg_n + ii*32 < n ? *(float4 *)&b[(ldg_n + ii*32) * k + ldg_k] : 0;
        }

        for (int ii = 0; ii < 2; ii++) {
            *(long *)&lds_a[0][sts_mn + ii*32][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_a[ii][0];
            *(long *)&lds_a[0][sts_mn + ii*32][SWZ(sts_mn, sts_k + 8)] = *(long *)&ldg_a[ii][8];
            *(long *)&lds_b[0][sts_mn + ii*32][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_b[ii][0];
            *(long *)&lds_b[0][sts_mn + ii*32][SWZ(sts_mn, sts_k + 8)] = *(long *)&ldg_b[ii][8];
        }

        __syncthreads();
    }

    for (int ib = 1; ib < k/BLOCK; ib++) {
        __hip_fp8_e4m3_fnuz ldg_a[2][16], ldg_b[2][16];
        for (int ii = 0; ii < 2; ii++) {
            *(float4 *)ldg_a[ii] = ldg_m + ii*32 < m ? *(float4 *)&a[(ldg_m + ii*32) * k + ldg_k + ib*BLOCK] : 0;
            *(float4 *)ldg_b[ii] = ldg_n + ii*32 < n ? *(float4 *)&b[(ldg_n + ii*32) * k + ldg_k + ib*BLOCK] : 0;
        }

        float this_as[2][4];
        for (int ii = 0; ii < 2; ii++) {
            *(float4 *)this_as[ii] = as_m + ii*16 < m ? *(float4 *)&as[(ib-1)*m + as_m + ii*16] : 0;
        }
        float this_bs = bs[(ib-1) * ((n+BLOCK-1)/BLOCK) + bs_n];

        float4 block_tile_d[2][2] = {0};
        long tile_a[2][4], tile_b[2][4];

        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_a[ii][kk] = *(long *)&lds_a[(ib-1)%2][lds_m + ii*16][SWZ(lds_m, lds_k + kk*32)];
            }
        }
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_b[jj][kk] = *(long *)&lds_b[(ib-1)%2][lds_n + jj*16][SWZ(lds_n, lds_k + kk*32)];
            }
        }
        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
                for (int kk = 0; kk < 4; kk++) {
                    block_tile_d[ii][jj] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(tile_a[ii][kk], tile_b[jj][kk], block_tile_d[ii][jj], 0, 0, 0);
                }
                for (int kk = 0; kk < 4; kk++) {
                    tile_d[ii][jj][kk] += block_tile_d[ii][jj][kk] * this_as[ii][kk] * this_bs;
                }
            }
        }

        for (int ii = 0; ii < 2; ii++) {
            *(long *)&lds_a[ib%2][sts_mn + ii*32][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_a[ii][0];
            *(long *)&lds_a[ib%2][sts_mn + ii*32][SWZ(sts_mn, sts_k + 8)] = *(long *)&ldg_a[ii][8];
            *(long *)&lds_b[ib%2][sts_mn + ii*32][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_b[ii][0];
            *(long *)&lds_b[ib%2][sts_mn + ii*32][SWZ(sts_mn, sts_k + 8)] = *(long *)&ldg_b[ii][8];
        }

        __syncthreads();
    }

    { // last COMPUTE
        float this_as[2][4];
        for (int ii = 0; ii < 2; ii++) {
            *(float4 *)this_as[ii] = as_m + ii*16 < m ? *(float4 *)&as[(k/BLOCK-1)*m + as_m + ii*16] : 0;
        }
        float this_bs = bs[(k/BLOCK-1) * ((n+BLOCK-1)/BLOCK) + bs_n];

        float4 block_tile_d[2][2] = {0};
        long tile_a[2][4], tile_b[2][4];
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_a[ii][kk] = *(long *)&lds_a[(k/BLOCK-1)%2][lds_m + ii*16][SWZ(lds_m, lds_k + kk*32)];
            }
        }
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_b[jj][kk] = *(long *)&lds_b[(k/BLOCK-1)%2][lds_n + jj*16][SWZ(lds_n, lds_k + kk*32)];
            }
        }

        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
                for (int kk = 0; kk < 4; kk++) {
                    block_tile_d[ii][jj] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(tile_a[ii][kk], tile_b[jj][kk], block_tile_d[ii][jj], 0, 0, 0);
                }
                for (int kk = 0; kk < 4; kk++) {
                    tile_d[ii][jj][kk] += block_tile_d[ii][jj][kk] * this_as[ii][kk] * this_bs;
                }
            }
        }
    }

    // finally, write the result as bf16
    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
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
    custom_kernel<<<dim3((m+63)/64, (n+63)/64), dim3(64, 4), 0, 0>>> ((__hip_fp8_e4m3_fnuz*)a.data_ptr(), (__hip_fp8_e4m3_fnuz*)b.data_ptr(),
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
    a = a.contiguous()
    b = b.contiguous()
    module.fp8_mm(a, b, a_scale, b_scale, c)
    return c