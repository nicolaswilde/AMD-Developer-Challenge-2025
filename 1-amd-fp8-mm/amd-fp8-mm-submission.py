# This script provides a template for using load_inline to run a HIP kernel for
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
CPP_WRAPPER = """
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c, torch::Tensor c_tmp);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr const int BLOCK = 128;

#define SWZ_32ft2(m, n) (((m) * 8 + ((m) / 16) * 8 + (n)) % 128)
__global__ void fp8_mm_32x32_16x16_ft2(
            const __hip_fp8_e4m3_fnuz* __restrict__ a,
            const __hip_fp8_e4m3_fnuz* __restrict__ b,
            const float* __restrict__ as,
            const float* __restrict__ bs,
            __hip_bfloat16* __restrict__ c, int m, int n, int k) {

    using float4 = float __attribute__((ext_vector_type(4)));

    __shared__ __hip_fp8_e4m3_fnuz lds_a[32][128];
    __shared__ __hip_fp8_e4m3_fnuz lds_b[32][128];

    int thread_block_m = blockIdx.x * 32;
    int thread_block_n = blockIdx.y * 32;

    int wave_m = (threadIdx.y / 2) * 16;
    int wave_n = (threadIdx.y % 2) * 16;

    int thread_c_m = (threadIdx.x / 16) * 4;
    int thread_c_n =  threadIdx.x % 16;

    // LDGSTS A&B Parameters
    int sts_k  = threadIdx.y * 32 + (threadIdx.x / 8) * 4;
    int sts_mn = (threadIdx.x % 8) * 4;
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

    float4 tile_d[1][1] = {0};

    for (int ib = 0; ib < k/BLOCK; ib++) {
        __hip_fp8_e4m3_fnuz ldg_a  [1][4][4], ldg_b  [1][4][4];
        __hip_fp8_e4m3_fnuz ldg_a_t[1][4][4], ldg_b_t[1][4][4];
        for (int ii = 0; ii < 1; ii++) {
            for (int jj = 0; jj < 4; jj++) {
                *(float *)ldg_a[ii][jj] = *(float *)&a[(ldg_k + ii*32 + jj + ib*BLOCK) * m + ldg_m];
                *(float *)ldg_b[ii][jj] = *(float *)&b[(ldg_k + ii*32 + jj + ib*BLOCK) * n + ldg_n];
            }
        }

        float this_as[1][4];
        for (int ii = 0; ii < 1; ii++) {
            *(float4 *)this_as[ii] = *(float4 *)&as[ib*m + as_m + ii*16];
        }
        float this_bs = bs[ib * ((n+BLOCK-1)/BLOCK) + bs_n];

        for (int ii = 0; ii < 1; ii++) {
            for (int jj = 0; jj < 4; jj++) {
                for (int kk = 0; kk < 4; kk++) {
                    ldg_a_t[ii][jj][kk] = ldg_a[ii][kk][jj];
                    ldg_b_t[ii][jj][kk] = ldg_b[ii][kk][jj];
                }
            }
            for (int jj = 0; jj < 4; jj++) {
                *(float *)&lds_a[sts_mn + jj][SWZ_32ft2(sts_mn + jj, sts_k + ii*64)] = *(float *)&ldg_a_t[ii][jj];
                *(float *)&lds_b[sts_mn + jj][SWZ_32ft2(sts_mn + jj, sts_k + ii*64)] = *(float *)&ldg_b_t[ii][jj];
            }
        }

        __syncthreads();

        float4 block_tile_d[1][1] = {0};
        long tile_a[1][4], tile_b[1][4];
        for (int ii = 0; ii < 1; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_a[ii][kk] = *(long *)&lds_a[lds_m + ii*16][SWZ_32ft2(lds_m + ii*16, lds_k + kk*32)];
            }
        }
        for (int jj = 0; jj < 1; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_b[jj][kk] = *(long *)&lds_b[lds_n + jj*16][SWZ_32ft2(lds_n + jj*16, lds_k + kk*32)];
            }
        }
        for (int ii = 0; ii < 1; ii++) {
            for (int jj = 0; jj < 1; jj++) {
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
    for (int ii = 0; ii < 1; ii++) {
        for (int jj = 0; jj < 1; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                c[(c_m + ii*16 + kk) * n + c_n + jj*16] = (__hip_bfloat16)(tile_d[ii][jj][kk]);
            }
        }
    }
}

__global__ void insitu_transpose_128x128(const __hip_fp8_e4m3_fnuz* __restrict__ a, int m, int n) {
    // transpose in every 128x128 block

    using float4 = float __attribute__((ext_vector_type(4)));

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int thread_m = threadIdx.x / 8;
    int thread_n = (threadIdx.x % 8) * 16;

    __shared__ __hip_fp8_e4m3_fnuz smem[128][129];

    __hip_fp8_e4m3_fnuz data[16];

    *(float4 *)&data[0] = *(float4 *)&a[(thread_block_m + thread_m) * n + thread_block_n + thread_n];

    for (int i = 0; i < 16; i++) {
        smem[thread_n + i][thread_m] = data[i];
        // smem[thread_n + i][((thread_n / 16) * 16 + thread_m) % 128] = data[i]; // swizzle
    }

    __syncthreads();

    *(float4 *)&a[(thread_block_m + thread_m) * n + thread_block_n + thread_n] = *(float4 *)&smem[thread_m][thread_n];
    // *(float4 *)&a[(thread_block_m + thread_m) * n + thread_block_n + thread_n] = *(float4 *)&smem[thread_m][((thread_m / 16) * 16 + thread_n) % 128]; // swizzle
}

#define SWZ_128pt(m, n) (((m) * 8 + (n)) % 128)
__launch_bounds__(512, 2)
__global__ void fp8_mm_128x128_pt(
        const __hip_fp8_e4m3_fnuz* __restrict__ a,
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
    int sts_mn = threadIdx.y * 8 + threadIdx.x / 8;
    int sts_k  = (threadIdx.x % 8) * 16;
    int ldg_m = thread_block_m + sts_k;
    int ldg_n = thread_block_n + sts_k;
    int ldg_k = sts_mn;

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
        __hip_fp8_e4m3_fnuz ldg_a[2][16], ldg_b[2][16];
        for (int ii = 0; ii < 2; ii++) {
            *(float4 *)ldg_a[ii] = *(float4 *)&a[(ldg_k + ib*BLOCK + ii*64) * m + ldg_m];
            *(float4 *)ldg_b[ii] = *(float4 *)&b[(ldg_k + ib*BLOCK + ii*64) * n + ldg_n];
        }

        float this_as[2][4];
        for (int ii = 0; ii < 2; ii++) {
            *(float4 *)this_as[ii] = *(float4 *)&as[ib*m + as_m + ii*16];
        }
        float this_bs = bs[ib * ((n+BLOCK-1)/BLOCK) + bs_n];

        for (int ii = 0; ii < 2; ii++) {
            *(long *)&lds_a[sts_mn + ii*64][SWZ_128pt(sts_mn, sts_k    )] = *(long *)&ldg_a[ii][0];
            *(long *)&lds_a[sts_mn + ii*64][SWZ_128pt(sts_mn, sts_k + 8)] = *(long *)&ldg_a[ii][8];
            *(long *)&lds_b[sts_mn + ii*64][SWZ_128pt(sts_mn, sts_k    )] = *(long *)&ldg_b[ii][0];
            *(long *)&lds_b[sts_mn + ii*64][SWZ_128pt(sts_mn, sts_k + 8)] = *(long *)&ldg_b[ii][8];
        }

        __syncthreads();

        float4 block_tile_d[2][4] = {0};
        long tile_a[2][4], tile_b[4][4];
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_a[ii][kk] = *(long *)&lds_a[lds_m + ii*16][SWZ_128pt(lds_m, lds_k + kk*32)];
            }
        }
        for (int jj = 0; jj < 4; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_b[jj][kk] = *(long *)&lds_b[lds_n + jj*16][SWZ_128pt(lds_n, lds_k + kk*32)];
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
                c[(c_m + ii*16 + kk) * n + c_n + jj*16] = (__hip_bfloat16)(tile_d[ii][jj][kk]);
            }
        }
    }
}

#define SWZ_128ft2(m, n) (((m) * 8 + ((m) / 16) * 8 + (n)) % 128)
__launch_bounds__(512, 2)
__global__ void fp8_mm_128x128_ft2(
        const __hip_fp8_e4m3_fnuz* __restrict__ a,
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
    int sts_k  = threadIdx.y * 8 + (threadIdx.x / 32) * 4;
    int sts_mn = (threadIdx.x % 32) * 4;
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
            for (int jj = 0; jj < 4; jj++) {
                *(float *)&lds_a[sts_mn + jj][SWZ_128ft2(sts_mn + jj, sts_k + ii*64)] = *(float *)&ldg_a_t[ii][jj];
                *(float *)&lds_b[sts_mn + jj][SWZ_128ft2(sts_mn + jj, sts_k + ii*64)] = *(float *)&ldg_b_t[ii][jj];
            }
        }

        __syncthreads();

        float4 block_tile_d[2][4] = {0};
        long tile_a[2][4], tile_b[4][4];
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_a[ii][kk] = *(long *)&lds_a[lds_m + ii*16][SWZ_128ft2(lds_m + ii*16, lds_k + kk*32)];
            }
        }
        for (int jj = 0; jj < 4; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_b[jj][kk] = *(long *)&lds_b[lds_n + jj*16][SWZ_128ft2(lds_n + jj*16, lds_k + kk*32)];
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

__launch_bounds__(512, 2)
__global__ void fp8_mm_128x128_ft2_split4(
        const __hip_fp8_e4m3_fnuz* __restrict__ a,
        const __hip_fp8_e4m3_fnuz* __restrict__ b,
        const float* __restrict__ as,
        const float* __restrict__ bs,
        float* __restrict__ c,
        int m, int n, int k) {

    using float4 = float __attribute__((ext_vector_type(4)));

    __shared__ __hip_fp8_e4m3_fnuz lds_a[128][128];
    __shared__ __hip_fp8_e4m3_fnuz lds_b[128][128];

    int split_idx = blockIdx.z;
    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int wave_m = (threadIdx.y / 2) * 32;
    int wave_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 16) * 4;
    int thread_c_n =  threadIdx.x % 16;

    // LDGSTS A&B Parameters
    int sts_k  = threadIdx.y * 8 + (threadIdx.x / 32) * 4;
    int sts_mn = (threadIdx.x % 32) * 4;
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

    float4 tile_d[2][4] = {0};

    for (int ib = split_idx * (k/BLOCK/4); ib < (split_idx + 1) * (k/BLOCK/4); ib++) {
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
            for (int jj = 0; jj < 4; jj++) {
                *(float *)&lds_a[sts_mn + jj][SWZ_128ft2(sts_mn + jj, sts_k + ii*64)] = *(float *)&ldg_a_t[ii][jj];
                *(float *)&lds_b[sts_mn + jj][SWZ_128ft2(sts_mn + jj, sts_k + ii*64)] = *(float *)&ldg_b_t[ii][jj];
            }
        }

        __syncthreads();

        float4 block_tile_d[2][4] = {0};
        long tile_a[2][4], tile_b[4][4];
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_a[ii][kk] = *(long *)&lds_a[lds_m + ii*16][SWZ_128ft2(lds_m + ii*16, lds_k + kk*32)];
            }
        }
        for (int jj = 0; jj < 4; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_b[jj][kk] = *(long *)&lds_b[lds_n + jj*16][SWZ_128ft2(lds_n + jj*16, lds_k + kk*32)];
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
    float *split_c = c + split_idx * (m * n);
    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 4; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                if (c_m + ii*16 < m && c_n + jj*16 < n) {
                    split_c[(c_m + ii*16 + kk) * n + c_n + jj*16] = (tile_d[ii][jj][kk]);
                }
            }
        }
    }
}
__global__ void fp8_mm_128x128_ft2_split4_merge(const float* __restrict__ c_split, __hip_bfloat16* __restrict__ c, int m, int n) {
    using float4 = float __attribute__((ext_vector_type(4)));
    int idx = blockIdx.x;
    for (int i = threadIdx.x * 8; i < n; i += blockDim.x * 8) {
        float4 val0[2], val1[2], val2[2], val3[2];
        val0[0] = *(float4 *)&c_split[idx * n + i];
        val0[1] = *(float4 *)&c_split[idx * n + i + 4];
        val1[0] = *(float4 *)&c_split[idx * n + i + m * n];
        val1[1] = *(float4 *)&c_split[idx * n + i + 4 + m * n];
        val2[0] = *(float4 *)&c_split[idx * n + i + 2 * m * n];
        val2[1] = *(float4 *)&c_split[idx * n + i + 4 + 2 * m * n];
        val3[0] = *(float4 *)&c_split[idx * n + i + 3 * m * n];
        val3[1] = *(float4 *)&c_split[idx * n + i + 4 + 3 * m * n];

        __hip_bfloat16 sum[8];
        sum[0] = val0[0][0] + val1[0][0] + val2[0][0] + val3[0][0];
        sum[1] = val0[0][1] + val1[0][1] + val2[0][1] + val3[0][1];
        sum[2] = val0[0][2] + val1[0][2] + val2[0][2] + val3[0][2];
        sum[3] = val0[0][3] + val1[0][3] + val2[0][3] + val3[0][3];
        sum[4] = val0[1][0] + val1[1][0] + val2[1][0] + val3[1][0];
        sum[5] = val0[1][1] + val1[1][1] + val2[1][1] + val3[1][1];
        sum[6] = val0[1][2] + val1[1][2] + val2[1][2] + val3[1][2];
        sum[7] = val0[1][3] + val1[1][3] + val2[1][3] + val3[1][3];

        *(float4 *)&c[idx * n + i] = *(float4 *)&sum[0];
    }
}


void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c, torch::Tensor c_tmp) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    __hip_fp8_e4m3_fnuz* a_ptr = (__hip_fp8_e4m3_fnuz*)a.data_ptr();
    __hip_fp8_e4m3_fnuz* b_ptr = (__hip_fp8_e4m3_fnuz*)b.data_ptr();
    __hip_bfloat16* c_ptr = (__hip_bfloat16*)c.data_ptr();
    float* c_tmp_ptr = (float*)c_tmp.data_ptr();
    float* as_ptr = (float*)as.data_ptr();
    float* bs_ptr = (float*)bs.data_ptr();
    // mmsplitk
    // 0. m: 1024; n: 1536; k: 7168;
    // 2. m: 1024; n: 576 ; k: 7168;
    // 7. m: 1024; n: 512 ; k: 7168;
    if (m == 1024 && n == 1536 && k == 7168 ||
        m == 1024 && n == 576  && k == 7168 ||
        m == 1024 && n == 512  && k == 7168) {
        fp8_mm_128x128_ft2_split4<<<dim3((m+127)/128, (n+127)/128, 4), dim3(64, 8), 0, 0>>> (a_ptr, b_ptr, as_ptr, bs_ptr, c_tmp_ptr, m, n, k);
        fp8_mm_128x128_ft2_split4_merge<<<dim3(m), dim3(64), 0, 0>>>(c_tmp_ptr, c_ptr, m, n);
        return;
    }
    // mm128x128ft2
    // 1. m: 1024; n: 3072; k: 1536;
    // 3. m: 1024; n: 7168; k: 256 ;
    // 4. m: 1024; n: 7168; k: 2048;
    // 5. m: 1024; n: 4608; k: 7168;
    // 8. m: 1024; n: 4096; k: 512 ;
    // 11. m: 6144; n: 576 ; k: 7168;
    // 16. m: 6144; n: 512 ; k: 7168;
    else if (m == 1024 && n == 3072 && k == 1536 ||
             m == 1024 && n == 7168 && k == 256  ||
             m == 1024 && n == 7168 && k == 2048 ||
             m == 1024 && n == 4608 && k == 7168 ||
             m == 1024 && n == 4096 && k == 512  ||
             m == 6144 && n == 576  && k == 7168 ||
             m == 6144 && n == 512  && k == 7168) {
        fp8_mm_128x128_ft2<<<dim3((m+127)/128, (n+127)/128), dim3(64, 8), 0, 0>>> (a_ptr, b_ptr, as_ptr, bs_ptr, c_ptr, m, n, k);
        return;
    }
    // mm128x128pt
    // 6. m: 1024; n: 7168; k: 2304;
    // 9. m: 6144; n: 1536; k: 7168;
    // 10. m: 6144; n: 3072; k: 1536;
    // 12. m: 6144; n: 7168; k: 256 ;
    // 13. m: 6144; n: 7168; k: 2048;
    // 14. m: 6144; n: 4608; k: 7168;
    // 15. m: 6144; n: 7168; k: 2304;
    // 17. m: 6144; n: 4096; k: 512 ;
    else if (m == 1024 && n == 7168 && k == 2304 ||
             m == 6144 && n == 1536 && k == 7168 ||
             m == 6144 && n == 3072 && k == 1536 ||
             m == 6144 && n == 7168 && k == 256  ||
             m == 6144 && n == 7168 && k == 2048 ||
             m == 6144 && n == 4608 && k == 7168 ||
             m == 6144 && n == 7168 && k == 2304 ||
             m == 6144 && n == 4096 && k == 512) {
        insitu_transpose_128x128<<<dim3((k+127)/128, (m+127)/128), dim3(1024), 0, 0>>>(a_ptr, k, m);
        insitu_transpose_128x128<<<dim3((k+127)/128, (n+127)/128), dim3(1024), 0, 0>>>(b_ptr, k, n);
        fp8_mm_128x128_pt<<<dim3((m+127)/128, (n+127)/128), dim3(64, 8), 0, 0>>> (a_ptr, b_ptr, as_ptr, bs_ptr, c_ptr, m, n, k);
        return;
    }
    else {
        fp8_mm_32x32_16x16_ft2<<<dim3((m+31)/32, (n+31)/32), dim3(64, 4), 0, 0>>> (a_ptr, b_ptr, as_ptr, bs_ptr, c_ptr, m, n, k);
        return;
    }
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
    m, n, k = a.shape[0], b.shape[0], a.shape[1]
    if (m == 1024 and n == 1536 and k == 7168 or
        m == 1024 and n == 576  and k == 7168 or
        m == 1024 and n == 512  and k == 7168):
        c_tmp = torch.empty(m * n * 4, dtype=torch.float32, device='cuda')
        module.fp8_mm(a, b, a_scale, b_scale, c, c_tmp)
    else:
        module.fp8_mm(a, b, a_scale, b_scale, c, c)
    return c