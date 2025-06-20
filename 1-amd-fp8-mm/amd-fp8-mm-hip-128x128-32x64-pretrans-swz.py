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

__global__ void fp8_mm_32x32(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs,
                   __hip_bfloat16* c, int m, int n, int k) {

    using float4 = float __attribute__((ext_vector_type(4)));

    __shared__ long lds_a[32][17];
    __shared__ long lds_b[32][17];
    __hip_fp8_e4m3_fnuz *lds_a_fp8 = reinterpret_cast<__hip_fp8_e4m3_fnuz *>(lds_a);
    __hip_fp8_e4m3_fnuz *lds_b_fp8 = reinterpret_cast<__hip_fp8_e4m3_fnuz *>(lds_b);

    int thread_block_m = blockIdx.x * 32;
    int thread_block_n = blockIdx.y * 32;

    int wave_m = (threadIdx.y / 2) * 16;
    int wave_n = (threadIdx.y % 2) * 16;

    int thread_c_m = (threadIdx.x / 16) * 4;
    int thread_c_n =  threadIdx.x % 16; // by float

    int cp_row = threadIdx.y * 32 + threadIdx.x / 2; // by fp8
    int cp_col = (threadIdx.x % 2) * 16;             // by fp8
    int cp_global_k = cp_row;
    int cp_global_m = cp_col + thread_block_m;       // by fp8
    int cp_global_n = cp_col + thread_block_n;       // by fp8

    int lds_shared_m = wave_m + threadIdx.x % 16;
    int lds_shared_n = wave_n + threadIdx.x % 16;
    int lds_shared_k = threadIdx.x / 16; // by long

    int c_m = thread_block_m + wave_m + thread_c_m;
    int c_n = thread_block_n + wave_n + thread_c_n;

    float4 *a_ptr = (float4 *)a;
    float4 *b_ptr = (float4 *)b;

    float4 tile_d = {0};

    int as_m =  thread_block_m + wave_m + thread_c_m;
    int bs_n = (thread_block_n + wave_n + thread_c_n) / BLOCK;
    float *as_ptr = (float*)&as[as_m];
    float *bs_ptr = (float*)&bs[bs_n];

    for (int ib = 0; ib < k; ib += BLOCK) {
        __hip_fp8_e4m3_fnuz ldg_a[16], ldg_b[16];
        *(float4 *)ldg_a = a_ptr[(cp_global_k + ib) * m/16 + cp_global_m/16];
        *(float4 *)ldg_b = b_ptr[(cp_global_k + ib) * n/16 + cp_global_n/16];
        float4 this_as = *(float4 *)(&as_ptr[0]);
        float  this_bs = bs_ptr[0];
        for (int ii = 0; ii < 16; ii++) {
            lds_a_fp8[(cp_col + ii) * 136 + cp_row] = ldg_a[ii];
            lds_b_fp8[(cp_col + ii) * 136 + cp_row] = ldg_b[ii];
        }
        __syncthreads();
        float4 block_tile_d = {0};
        block_tile_d = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(lds_a[lds_shared_m][lds_shared_k +  0], lds_b[lds_shared_n][lds_shared_k +  0], block_tile_d, 0, 0, 0);
        block_tile_d = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(lds_a[lds_shared_m][lds_shared_k +  4], lds_b[lds_shared_n][lds_shared_k +  4], block_tile_d, 0, 0, 0);
        block_tile_d = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(lds_a[lds_shared_m][lds_shared_k +  8], lds_b[lds_shared_n][lds_shared_k +  8], block_tile_d, 0, 0, 0);
        block_tile_d = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(lds_a[lds_shared_m][lds_shared_k + 12], lds_b[lds_shared_n][lds_shared_k + 12], block_tile_d, 0, 0, 0);
        __syncthreads();
        tile_d[0] += block_tile_d[0] * this_as[0] * this_bs;
        tile_d[1] += block_tile_d[1] * this_as[1] * this_bs;
        tile_d[2] += block_tile_d[2] * this_as[2] * this_bs;
        tile_d[3] += block_tile_d[3] * this_as[3] * this_bs;
        as_ptr += m;
        bs_ptr += (n+BLOCK-1)/BLOCK;
    }

    // finally, write the result as bf16
    c[(c_m + 0) * n + c_n] = (__hip_bfloat16)(tile_d[0]);
    c[(c_m + 1) * n + c_n] = (__hip_bfloat16)(tile_d[1]);
    c[(c_m + 2) * n + c_n] = (__hip_bfloat16)(tile_d[2]);
    c[(c_m + 3) * n + c_n] = (__hip_bfloat16)(tile_d[3]);
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

__launch_bounds__(512, 2)
__global__ void fp8_mm_128x128(const __hip_fp8_e4m3_fnuz* __restrict__ a,
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
            *(long *)&lds_a[sts_mn + ii*64][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_a[ii][0];
            *(long *)&lds_a[sts_mn + ii*64][SWZ(sts_mn, sts_k + 8)] = *(long *)&ldg_a[ii][8];
            *(long *)&lds_b[sts_mn + ii*64][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_b[ii][0];
            *(long *)&lds_b[sts_mn + ii*64][SWZ(sts_mn, sts_k + 8)] = *(long *)&ldg_b[ii][8];
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
                c[(c_m + ii*16 + kk) * n + c_n + jj*16] = (__hip_bfloat16)(tile_d[ii][jj][kk]);
            }
        }
    }
}

void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    if (m % 128 != 0 || n % 128 != 0) {
        fp8_mm_32x32<<<dim3((m+31)/32, (n+31)/32), dim3(64, 4), 0, 0>>> (
            (__hip_fp8_e4m3_fnuz*)a.data_ptr(), (__hip_fp8_e4m3_fnuz*)b.data_ptr(),
            as.data_ptr<float>(), bs.data_ptr<float>(), (__hip_bfloat16*)c.data_ptr(), m, n, k);
    }
    else {
        insitu_transpose_128x128<<<dim3((k+127)/128, (m+127)/128), dim3(1024), 0, 0>>>((__hip_fp8_e4m3_fnuz*)a.data_ptr(), k, m);
        insitu_transpose_128x128<<<dim3((k+127)/128, (n+127)/128), dim3(1024), 0, 0>>>((__hip_fp8_e4m3_fnuz*)b.data_ptr(), k, n);
        fp8_mm_128x128<<<dim3((m+127)/128, (n+127)/128), dim3(64, 8), 0, 0>>> (
            (__hip_fp8_e4m3_fnuz*)a.data_ptr(), (__hip_fp8_e4m3_fnuz*)b.data_ptr(),
            as.data_ptr<float>(), bs.data_ptr<float>(), (__hip_bfloat16*)c.data_ptr(), m, n, k);
        // insitu_transpose_128x128<<<dim3((k+127)/128, (m+127)/128), dim3(64), 0, 0>>>((__hip_fp8_e4m3_fnuz*)a.data_ptr(), k, m);
        // insitu_transpose_128x128<<<dim3((k+127)/128, (n+127)/128), dim3(64), 0, 0>>>((__hip_fp8_e4m3_fnuz*)b.data_ptr(), k, n);
    }
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