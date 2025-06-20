# This script provides a template for using load_inline to run a HIP kernel for
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
CPP_WRAPPER = """
void moe(torch::Tensor t_input, torch::Tensor t_router,
         torch::Tensor t_gate_T, torch::Tensor t_up_T, torch::Tensor t_down_T,
         torch::Tensor t_gate, torch::Tensor t_up, torch::Tensor t_down,
         torch::Tensor t_output, int dh, int de, int nre, int nse, int ept, int n,
         torch::Tensor t_score_fp32, torch::Tensor t_score_fp16,
         torch::Tensor t_top_indices, torch::Tensor t_token_count,
         torch::Tensor t_index_in_expert,
         torch::Tensor t_expert_offset, torch::Tensor t_input_reorder,
         torch::Tensor t_hidden, torch::Tensor t_output_down);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define SWZ(m, n) (((m) * 4 + (n)) % 64) // by half

using float16 = float __attribute__((ext_vector_type(16)));
typedef struct {
    float16 val[2][2];
} float16_2x2_t;
using half4 = __fp16 __attribute__((ext_vector_type(4)));
typedef struct {
    half4 x;
    half4 y;
} half4x2;

__global__ void softmax_fp32(const float *input, float *output, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        float max_val = input[idx * n];
        for (int i = 1; i < n; i++) {
            float val = input[idx * n + i];
            max_val = val > max_val ? val : max_val;
        }
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float exp_val = __expf(input[idx * n + i] - max_val);
            output[idx * n + i] = exp_val;
            sum += exp_val;
        }
        for (int i = 0; i < n; i++) {
            output[idx * n + i] = output[idx * n + i] / sum;
        }
    }
}

__global__ void softmax_fp32_fp16(const float *input, __half *output, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        __half max_val = __float2half(-INFINITY);
        for (int i = 0; i < n; i++) {
            __half val = __float2half(input[idx * n + i]);
            output[idx * n + i] = val;
            max_val = __hgt(val, max_val) ? val : max_val;
        }
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float exp_val = __expf(__half2float(output[idx * n + i]) - __half2float(max_val));
            sum += exp_val;
        }
        for (int i = 0; i < n; i++) {
            float exp_val = __expf(__half2float(output[idx * n + i]) - __half2float(max_val));
            output[idx * n + i] = __float2half(exp_val / sum);
        }
    }
}

__global__ void topk(const __half *input, int *top_indices, int m, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        // Find top-k indices for row idx
        // Simple selection sort for small k
        for (int i = 0; i < k; ++i) {
            __half max_val = __float2half(-INFINITY);
            int max_idx = -1;
            for (int j = 0; j < n; ++j) {
                if (__hgt(input[idx * n + j], max_val)) {
                    int found = 0;
                    for (int l = 0; l < i; ++l) {
                        if (top_indices[idx * k + l] == j) {
                            found = 1;
                            break;
                        }
                    }
                    if (found == 0) {
                        max_val = input[idx * n + j];
                        max_idx = j;
                    }
                }
            }
            top_indices[idx * k + i] = max_idx;
        }
    }
}

__global__ void calculate_index(
    const int *top_indices, int *token_count, int *index_in_expert, int n, int ept, int nre) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int j = 0; j < ept; ++j) {
            int expert = top_indices[idx * ept + j];
            int offset = atomicAdd(&token_count[expert], 1);
            index_in_expert[idx * ept + j] = offset;
        }
    }
}

__global__ void reorder_input(__half *input, int *top_indices, int *expert_offset, int *index_in_expert, __half *input_reorder, int n, int ept, int nre, int dh) {
    int idx = blockIdx.x;
    if (idx < n) {
        for (int i = 0; i < ept; ++i) {
            int expert_id = top_indices[idx * ept + i];
            int token_location = expert_offset[expert_id] + index_in_expert[idx * ept + i];
            for (int j = threadIdx.x; j < dh; j += 64) {
                input_reorder[token_location * dh + j] = input[idx * dh + j];
            }
        }
    }
}

__global__ void pack_output(half *output_down, __half *output_ptr, const __half *score, int *top_indices, int *token_count, int *expert_offset, int *index_in_expert, int n, int ept, int nre, int nse, int dh) {
    int idx = blockIdx.x;
    if (idx < n) {
        for (int j = threadIdx.x; j < dh; j += 64) {
            float result = 0.0f;
            for (int i = 0; i < ept; ++i) {
                int expert_id = top_indices[idx * ept + i];
                int token_location = expert_offset[expert_id] + index_in_expert[idx * ept + i];
                result += __half2float(score[idx * nre + expert_id]) * __half2float(output_down[token_location * dh + j]);
            }
            for (int i = 0; i < nse; ++i) {
                result += __half2float(output_down[(n * (ept + i) + idx) * dh + j]);
            }
            output_ptr[idx * dh + j] = __float2half(result);
        }
    }
}

__global__ void transpose_expert_fp16_64x64(
        int n_experts, int m, int n, half* __restrict__ dst, half* __restrict__ src) {

    using half8 = __fp16 __attribute__((ext_vector_type(8)));

    int thread_block_e = blockIdx.z;
    int thread_block_m = blockIdx.x * 64;
    int thread_block_n = blockIdx.y * 64;

    int thread_m = threadIdx.x / 8;
    int thread_n = (threadIdx.x % 8) * 8;

    __shared__ char smem[64][129];

    half8 data[8];
    for (int i = 0; i < 8; i++) {
        data[i] = *(half8 *)&src[thread_block_e * m * n +
            (thread_block_m + thread_m + i*8) * n + thread_block_n + thread_n];
        for (int j = 0; j < 8; j++) {
            *(__fp16 *)&smem[thread_n + j][(thread_m + 8*i) * 2] = data[i][j];
        }
    }

    __syncthreads();

    for (int i = 0; i < 8; i++) {
        *(half8 *)&dst[thread_block_e * m * n +
            (thread_block_n + thread_m + i*8) * m + thread_block_m + thread_n] =
            *(half8 *)&smem[thread_m + 8*i][thread_n*2];
    }
}

__device__ __inline__ float16_2x2_t hgemm_128x128x64_clean(
    const half *a, const half *b, int M, int N, int K,
    __shared__ half s_a[128][64], __shared__ half s_b[128][64]) {

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int warp_m = (threadIdx.y / 2) * 64;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 32) * 4;
    int thread_c_n = threadIdx.x % 32;

    // LDGSTS A&B Parameters
    int sts_mn = threadIdx.y * 8 + threadIdx.x / 8;
    int sts_k = (threadIdx.x % 8) * 8; // by half
    int ldg_m = thread_block_m + sts_mn;
    int ldg_n = thread_block_n + sts_mn;
    int ldg_k = sts_k;

    // LDS A&B Parameters
    int lds_m = warp_m + threadIdx.x % 32;
    int lds_n = warp_n + threadIdx.x % 32;
    int lds_k = (threadIdx.x / 32) * 4; // by half

    // C&D Parameters
    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    float16_2x2_t tile_d = {0};

    for (int ib = 0; ib < K/64; ib++) {
        half4x2 ldg_a[4], ldg_b[4];
        for (int ii = 0; ii < 4; ii++) {
            // ldg_a[ii] = *(half4x2 *)&a[OFFSET(ldg_m + ii*32, ldg_k + ib*64, K)];
            // ldg_b[ii] = *(half4x2 *)&b[OFFSET(ldg_n + ii*32, ldg_k + ib*64, K)];
            ldg_a[ii] = ldg_m + ii*32 < M ? *(half4x2 *)&a[OFFSET(ldg_m + ii*32, ldg_k + ib*64, K)] : half4x2{0, 0};
            ldg_b[ii] = ldg_n + ii*32 < N ? *(half4x2 *)&b[OFFSET(ldg_n + ii*32, ldg_k + ib*64, K)] : half4x2{0, 0};
        }

        for (int ii = 0; ii < 4; ii++) {
            *(half4 *)&s_a[sts_mn + ii*32][SWZ(sts_mn, sts_k    )] = ldg_a[ii].x;
            *(half4 *)&s_a[sts_mn + ii*32][SWZ(sts_mn, sts_k + 4)] = ldg_a[ii].y;
            *(half4 *)&s_b[sts_mn + ii*32][SWZ(sts_mn, sts_k    )] = ldg_b[ii].x;
            *(half4 *)&s_b[sts_mn + ii*32][SWZ(sts_mn, sts_k + 4)] = ldg_b[ii].y;
        }

        __syncthreads();

        half4 tile_a[2][8], tile_b[2][8];
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 8; kk++) {
                tile_a[ii][kk] = *(half4 *)&s_a[lds_m + ii*32][SWZ(lds_m, lds_k + kk*8)];
            }
        }
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 8; kk++) {
                tile_b[ii][kk] = *(half4 *)&s_b[lds_n + ii*32][SWZ(lds_n, lds_k + kk*8)];
            }
        }
        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
                for (int kk = 0; kk < 8; kk++) {
                    tile_d.val[ii][jj] = __builtin_amdgcn_mfma_f32_32x32x8f16(tile_a[ii][kk], tile_b[jj][kk], tile_d.val[ii][jj], 0, 0, 0);
                }
            }
        }

        __syncthreads();
    }

    return tile_d;
}

__launch_bounds__(256, 2)
__global__ void hgemm_128x128x64(
    const half *a, const half *b, float *c, int M, int N, int K) {

    __shared__ half s_a[128][64];
    __shared__ half s_b[128][64];

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int warp_m = (threadIdx.y / 2) * 64;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 32) * 4;
    int thread_c_n = threadIdx.x % 32;

    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    float16_2x2_t tile_d = hgemm_128x128x64_clean( a, b, M, N, K, s_a, s_b);

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                for (int ll = 0; ll < 4; ll++) {
                    if (c_m + ii*32 + kk*8 + ll < M && c_n + jj*32 < N)
                        c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32] = tile_d.val[ii][jj][kk*4 + ll];
                }
            }
        }
    }
}

__launch_bounds__(256, 2)
__global__ void moe_hgemm_128x128x64(
    const half *a_ptr, const half *b_ptr, half *c_ptr, int nre, int *token_count, int *expert_offset, int N, int K) {

    const int M = token_count[blockIdx.z];

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;
    if (thread_block_m >= M) return;

    const half *a = a_ptr + expert_offset[blockIdx.z] * K;
    const half *b = b_ptr + blockIdx.z * N * K;
    half *c = c_ptr + expert_offset[blockIdx.z] * N;

    __shared__ half s_a[128][64];
    __shared__ half s_b[128][64];

    int warp_m = (threadIdx.y / 2) * 64;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 32) * 4;
    int thread_c_n = threadIdx.x % 32;

    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    float16_2x2_t tile_d = hgemm_128x128x64_clean(a, b, M, N, K, s_a, s_b);

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                for (int ll = 0; ll < 4; ll++) {
                    if (c_m + ii*32 + kk*8 + ll < M && c_n + jj*32 < N)
                        c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32] = __float2half(tile_d.val[ii][jj][kk*4 + ll]);
                }
            }
        }
    }
}

__launch_bounds__(256, 2)
__global__ void batched_hgemm_128x128x64(
    const half *a_ptr, const half *b_ptr, half *c_ptr, int bs, int M, int N, int K) {

    const half *a = a_ptr + blockIdx.z * M * K;
    const half *b = b_ptr + blockIdx.z * N * K;
    half *c = c_ptr + blockIdx.z * M * N;

    __shared__ half s_a[128][64];
    __shared__ half s_b[128][64];

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int warp_m = (threadIdx.y / 2) * 64;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 32) * 4;
    int thread_c_n = threadIdx.x % 32;

    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    float16_2x2_t tile_d = hgemm_128x128x64_clean(a, b, M, N, K, s_a, s_b);

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                for (int ll = 0; ll < 4; ll++) {
                    if (c_m + ii*32 + kk*8 + ll < M && c_n + jj*32 < N)
                        c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32] = __float2half(tile_d.val[ii][jj][kk*4 + ll]);
                }
            }
        }
    }
}

__launch_bounds__(256, 2)
__global__ void moe_hgemm_128x128x64_silu_mul(
    const half *a_ptr, const half *b_ptr, half *c_ptr, int nre, int *token_count, int *expert_offset, int N, int K) {

    const int M = token_count[blockIdx.z];

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;
    if (thread_block_m >= M) return;

    const half *a = a_ptr + expert_offset[blockIdx.z] * K;
    const half *b = b_ptr + blockIdx.z * N * K;
    half *c = c_ptr + expert_offset[blockIdx.z] * N;

    __shared__ half s_a[128][64];
    __shared__ half s_b[128][64];

    int warp_m = (threadIdx.y / 2) * 64;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 32) * 4;
    int thread_c_n = threadIdx.x % 32;

    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    float16_2x2_t tile_d = hgemm_128x128x64_clean(a, b, M, N, K, s_a, s_b);

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                for (int ll = 0; ll < 4; ll++) {
                    if (c_m + ii*32 + kk*8 + ll < M && c_n + jj*32 < N) {
                        float ori = __half2float(c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32]);
                        float val = tile_d.val[ii][jj][kk*4 + ll];
                        float silu_val = val * (1.0f / (1.0f + __expf(-val)));
                        c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32] = __float2half(ori * silu_val);
                    }
                }
            }
        }
    }
}

__launch_bounds__(256, 2)
__global__ void batched_hgemm_128x128x64_silu_mul(
    const half *a_ptr, const half *b_ptr, half *c_ptr, int bs, int M, int N, int K) {

    const half *a = a_ptr + blockIdx.z * M * K;
    const half *b = b_ptr + blockIdx.z * N * K;
    half *c = c_ptr + blockIdx.z * M * N;

    __shared__ half s_a[128][64];
    __shared__ half s_b[128][64];

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int warp_m = (threadIdx.y / 2) * 64;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 32) * 4;
    int thread_c_n = threadIdx.x % 32;

    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    float16_2x2_t tile_d = hgemm_128x128x64_clean(a, b, M, N, K, s_a, s_b);

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                for (int ll = 0; ll < 4; ll++) {
                    if (c_m + ii*32 + kk*8 + ll < M && c_n + jj*32 < N) {
                        float ori = __half2float(c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32]);
                        float val = tile_d.val[ii][jj][kk*4 + ll];
                        float silu_val = val * (1.0f / (1.0f + __expf(-val)));
                        c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32] = __float2half(ori * silu_val);
                    }
                }
            }
        }
    }
}

void moe(torch::Tensor t_input, torch::Tensor t_router,
         torch::Tensor t_gate_T, torch::Tensor t_up_T, torch::Tensor t_down_T,
         torch::Tensor t_gate, torch::Tensor t_up, torch::Tensor t_down,
         torch::Tensor t_output, int dh, int de, int nre, int nse, int ept, int n,
         torch::Tensor t_score_fp32, torch::Tensor t_score_fp16,
         torch::Tensor t_top_indices, torch::Tensor t_token_count,
         torch::Tensor t_index_in_expert,
         torch::Tensor t_expert_offset, torch::Tensor t_input_reorder,
         torch::Tensor t_hidden, torch::Tensor t_output_down) {

    if (dh % 128 != 0 || de % 128 != 0) return;

    using uint64_t = unsigned long long;

    __half *input_ptr   = (__half *)t_input.data_ptr();            // [n, dh]
    __half *router_ptr  = (__half *)t_router.data_ptr();           // [nre, dh]
    __half *gate_T_ptr  = (__half *)t_gate_T.data_ptr();           // [nre + nse, dh, de]
    __half *up_T_ptr    = (__half *)t_up_T.data_ptr();             // [nre + nse, dh, de]
    __half *down_T_ptr  = (__half *)t_down_T.data_ptr();           // [nre + nse, de, dh]
    __half *gate_ptr    = (__half *)t_gate.data_ptr();             // [nre + nse, de, dh]
    __half *up_ptr      = (__half *)t_up.data_ptr();               // [nre + nse, de, dh]
    __half *down_ptr    = (__half *)t_down.data_ptr();             // [nre + nse, dh, de]
    __half *output_ptr  = (__half *)t_output.data_ptr();           // [n, dh]

    float *score_fp32 = (float *)t_score_fp32.data_ptr();         // [n, nre]
    __half *score_fp16 = (__half *)t_score_fp16.data_ptr();       // [n, nre]
    int *top_indices = (int *)t_top_indices.data_ptr();           // [n, ept]
    int *token_count = (int *)t_token_count.data_ptr();           // [nre]
    int *index_in_expert = (int *)t_index_in_expert.data_ptr();   // [n, ept]
    int *expert_offset = (int *)t_expert_offset.data_ptr();       // [nre]
    __half *input_reorder = (__half *)t_input_reorder.data_ptr(); // [n * ept, dh]
    __half *hidden = (__half *)t_hidden.data_ptr();               // [n * (ept + nse), de]
    __half *output_down = (__half *)t_output_down.data_ptr();     // [n * (ept + nse), dh]

    // transpose
    transpose_expert_fp16_64x64<<<dim3(dh/64, de/64, nre + nse), dim3(64)>>>(nre + nse, dh, de, gate_ptr, gate_T_ptr);
    transpose_expert_fp16_64x64<<<dim3(dh/64, de/64, nre + nse), dim3(64)>>>(nre + nse, dh, de, up_ptr  , up_T_ptr  );
    transpose_expert_fp16_64x64<<<dim3(de/64, dh/64, nre + nse), dim3(64)>>>(nre + nse, de, dh, down_ptr, down_T_ptr);

    // expert score
    // float *score_fp32; // [n, nre]
    hgemm_128x128x64<<<dim3((n+127)/128, (nre+127)/128), dim3(64, 4)>>>
        (input_ptr, router_ptr, score_fp32, n, nre, dh);
    // __half *score_fp16; // [n, nre]
    softmax_fp32_fp16<<<dim3((n+63)/64), dim3(64)>>>(score_fp32, score_fp16, n, nre);


    // int *top_indices; // [n, ept], topk score
    topk<<<dim3((n+63)/64), dim3(64)>>>(score_fp16, top_indices, n, nre, ept);

    // int *token_count; // [nre], token count of each expert
    // int *index_in_expert; // [n, ept], index in expert of each token, i -> expert token index
    hipMemset(token_count, 0, nre * sizeof(int));
    calculate_index<<<dim3((n+63)/64), dim3(64)>>>(top_indices, token_count, index_in_expert, n, ept, nre);
    hipDeviceSynchronize(); // because token_count is used by host code

    // expert offset
    // int *expert_offset; // [nre]
    {
        int offset = 0;
        for (int i = 0; i < nre; i++) {
            expert_offset[i] = offset;
            offset += token_count[i];
        }
    }

    // reorder input
    // __half *input_reorder; // [n * ept, dh]
    reorder_input<<<dim3(n), dim3(64)>>>(input_ptr, top_indices, expert_offset, index_in_expert, input_reorder, n, ept, nre, dh);

    int max_token_count = 0;
    for (int i = 0; i < nre; i++) {
        if (token_count[i] > max_token_count) max_token_count = token_count[i];
    }

    // hidden state: silu(output_gate) * output_up
    // __half *hidden; // [n * (ept + nse), de]
    moe_hgemm_128x128x64<<<dim3((max_token_count+127)/128, de/128, nre), dim3(64, 4)>>>
        (input_reorder, up_ptr, hidden, nre, token_count, expert_offset, de, dh);
    batched_hgemm_128x128x64<<<dim3((n+127)/128, de/128, nse), dim3(64, 4)>>>
        (input_ptr, up_ptr + nre * dh * de, hidden + n * ept * de, nse, n, de, dh);
    moe_hgemm_128x128x64_silu_mul<<<dim3((max_token_count+127)/128, de/128, nre), dim3(64, 4)>>>
        (input_reorder, gate_ptr, hidden, nre, token_count, expert_offset, de, dh);
    batched_hgemm_128x128x64_silu_mul<<<dim3((n+127)/128, de/128, nse), dim3(64, 4)>>>
        (input_ptr, gate_ptr + nre * dh * de, hidden + n * ept * de, nse, n, de, dh);

    // down output
    // half *output_down; // [n * (ept + nse), dh]
    moe_hgemm_128x128x64<<<dim3((max_token_count+127)/128, dh/128, nre), dim3(64, 4)>>>
        (hidden, down_ptr, output_down, nre, token_count, expert_offset, dh, de);
    batched_hgemm_128x128x64<<<dim3((n+127)/128, dh/128, nse), dim3(64, 4)>>>
        (hidden + n * ept * de, down_ptr + nre * de * dh, output_down + n * ept * dh, nse, n, dh, de);

    // pack output
    pack_output<<<dim3(n), dim3(64)>>>(output_down, output_ptr, score_fp16, top_indices, token_count, expert_offset, index_in_expert, n, ept, nre, nse, dh);
}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='moe',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['moe'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O3", "-w"],
)

def custom_kernel(data: input_t) -> output_t:
    input_tensor, weights, config = data

    dh = config["d_hidden"]
    de = config["d_expert"]
    nre = config["n_routed_experts"]
    nse = config["n_shared_experts"]
    ept = config["n_experts_per_token"]
    bs = config["batch_size"]
    sl = config["seq_len"]

    input  = input_tensor                                                       # [bs, sl, dh]
    router = weights['router.weight']                                           # [nre, dh]
    gate_T = torch.empty(nre + nse, dh, de, dtype=torch.float16, device='cuda') # [nre + nse, dh, de]
    up_T   = torch.empty(nre + nse, dh, de, dtype=torch.float16, device='cuda') # [nre + nse, dh, de]
    down_T = torch.empty(nre + nse, de, dh, dtype=torch.float16, device='cuda') # [nre + nse, de, dh]
    gate   = torch.empty(nre + nse, de, dh, dtype=torch.float16, device='cuda') # [nre + nse, de, dh]
    up     = torch.empty(nre + nse, de, dh, dtype=torch.float16, device='cuda') # [nre + nse, de, dh]
    down   = torch.empty(nre + nse, dh, de, dtype=torch.float16, device='cuda') # [nre + nse, dh, de]
    output = torch.zeros(bs, sl, dh, dtype=torch.float16, device='cuda')        # [bs, sl, dh]
    for i in range(nre):
        gate_T[i] = weights[f'experts.{i}.0.weight']
        up_T[i] = weights[f'experts.{i}.1.weight']
        down_T[i] = weights[f'experts.{i}.2.weight']
    for i in range(nse):
        gate_T[nre + i] = weights['shared_experts.0.weight'][:, i * de:(i + 1) * de]
        up_T[nre + i] = weights['shared_experts.1.weight'][:, i * de:(i + 1) * de]
        down_T[nre + i] = weights['shared_experts.2.weight'][i * de:(i + 1) * de, :]

    score_fp32 = torch.empty(bs * sl * nre, dtype=torch.float32, device='cuda')                # [bs * sl, nre]
    score_fp16 = torch.empty(bs * sl * nre, dtype=torch.float16, device='cuda')                # [bs * sl, nre]
    top_indices = torch.empty(bs * sl * ept, dtype=torch.int32, device='cuda')                 # [bs * sl, ept]
    token_count = torch.empty(nre, dtype=torch.int32, device='cuda')                           # [nre]
    index_in_expert = torch.empty(bs * sl * ept, dtype=torch.int32, device='cuda')             # [bs * sl, ept]
    expert_offset = torch.empty(nre, dtype=torch.int32, device='cuda')                         # [nre]
    input_reorder = torch.empty(bs * sl * ept * dh, dtype=torch.float16, device='cuda')        # [bs * sl * ept, dh]
    hidden = torch.empty(bs * sl * (nre + nse) * de, dtype=torch.float16, device='cuda')       # [bs * sl * (nre + nse), de]
    output_down = torch.empty(bs * sl * (nre + nse) * dh, dtype=torch.float16, device='cuda')  # [bs * sl * (nre + nse), dh]

    module.moe(input, router, gate_T, up_T, down_T, gate, up, down, output,
        dh, de, nre, nse, ept, bs*sl,
        score_fp32, score_fp16, top_indices, token_count, index_in_expert, expert_offset,
        input_reorder, hidden, output_down)
    return output