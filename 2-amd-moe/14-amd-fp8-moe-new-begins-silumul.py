# This script provides a template for using load_inline to run a HIP kernel for
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
CPP_WRAPPER = """
int moe1(torch::Tensor t_input, torch::Tensor t_output,
         int dh, int de, int nre, int nse, int ept, int n,
         torch::Tensor t_score, torch::Tensor t_top_indices,
         torch::Tensor t_token_count, torch::Tensor t_index_in_expert,
         torch::Tensor t_input_reorder);
void pack(torch::Tensor t_input, torch::Tensor t_output,
          int dh, int de, int nre, int nse, int ept, int n,
          torch::Tensor t_score, torch::Tensor t_top_indices,
          torch::Tensor t_token_count, torch::Tensor t_index_in_expert,
          torch::Tensor t_output_down_re, torch::Tensor t_output_down_se, int max_count);
void re_silu_mul(torch::Tensor t_gate, torch::Tensor t_up, int ne, int n, int d, torch::Tensor t_token_count);
void se_silu_mul(torch::Tensor t_gate, torch::Tensor t_up, int ne, int n, int d);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>

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

__global__ void top4_nre32(const __half *input, int *top_indices, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m) return;

    using float4 = float __attribute__((ext_vector_type(4)));
    __half row[32];
    *(float4 *)&row[0] = *(float4 *)&input[idx * 32 + 0];
    *(float4 *)&row[8] = *(float4 *)&input[idx * 32 + 8];
    *(float4 *)&row[16] = *(float4 *)&input[idx * 32 + 16];
    *(float4 *)&row[24] = *(float4 *)&input[idx * 32 + 24];

    int top_idx[4] = {-1, -1, -1, -1};
    __half top_val[4] = {__float2half(-INFINITY), __float2half(-INFINITY), __float2half(-INFINITY), __float2half(-INFINITY)};

    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        if (__hgt(row[j], top_val[0])) {
            top_val[0] = row[j];
            top_idx[0] = j;
        }
    }

    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        if (j == top_idx[0]) continue;
        if (__hgt(row[j], top_val[1])) {
            top_val[1] = row[j];
            top_idx[1] = j;
        }
    }

    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        if (j == top_idx[0] || j == top_idx[1]) continue;
        if (__hgt(row[j], top_val[2])) {
            top_val[2] = row[j];
            top_idx[2] = j;
        }
    }

    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        if (j == top_idx[0] || j == top_idx[1] || j == top_idx[2]) continue;
        if (__hgt(row[j], top_val[3])) {
            top_val[3] = row[j];
            top_idx[3] = j;
        }
    }

    *(float4 *)&top_indices[idx * 4] = *(float4 *)&top_idx[0];
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

__global__ void calculate_index_nre32_ept4(
    const int *top_indices, int *token_count, int *index_in_expert, int n) {

    using float4 = float __attribute__((ext_vector_type(4)));

    const int NRE = 32;
    const int EPT = 4;
    __shared__ int block_count[NRE];
    __shared__ int block_offset[NRE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (threadIdx.x < NRE) {
        block_count[threadIdx.x] = 0;
    }

    int thread_offset[EPT];
    int thread_expert[EPT];
    *(float4 *)&thread_expert[0] = *(float4 *)&top_indices[idx * EPT + 0];
    for (int i = 0; i < EPT; i++) {
        thread_offset[i] = atomicAdd(&block_count[thread_expert[i]], 1);
    }

    if (threadIdx.x < NRE) {
        block_offset[threadIdx.x] = atomicAdd(&token_count[threadIdx.x], block_count[threadIdx.x]);
    }

    for (int i = 0; i < EPT; i++) {
        thread_offset[i] += block_offset[thread_expert[i]];
    }

    *(float4 *)&index_in_expert[idx * EPT + 0] = *(float4 *)&thread_offset[0];
}

__global__ void reorder_input(__half *input, int *top_indices, int *index_in_expert, __half *input_reorder, int n, int ept, int nre, int dh) {
    int idx = blockIdx.x;
    using float4 = float __attribute__((ext_vector_type(4)));
    if (idx < n) {
        for (int i = 0; i < ept; ++i) {
            int expert_id = top_indices[idx * ept + i];
            int token_location = expert_id * n + index_in_expert[idx * ept + i];
            #pragma unroll
            for (int j = threadIdx.x * 8; j < dh; j += 64 * 8) {
                float4 val = *(float4 *)&input[idx * dh + j];
                *(float4 *)&input_reorder[token_location * dh + j] = val;
                // input_reorder[token_location * dh + j] = input[idx * dh + j];
            }
        }
    }
}

__global__ void reorder_input_ept4nre32dh7168(__half *input, int *top_indices, int *index_in_expert, __half *input_reorder, int n, int ept, int nre, int dh) {
    int idx = blockIdx.x;
    using float4 = float __attribute__((ext_vector_type(4)));
    float4 val[14];
    #pragma unroll
    for (int i = 0; i < 14; i++) {
        val[i] = *(float4 *)&input[blockIdx.x * 7168 + threadIdx.x * 8 + i * 512];
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int expert_id = top_indices[idx * 4 + i];
        int token_location = expert_id * n + index_in_expert[idx * 4 + i];
        #pragma unroll
        for (int j = 0; j < 14; j++) {
            *(float4 *)&input_reorder[token_location * 7168 + threadIdx.x * 8 + j * 512] = val[j];
        }
    }
}

int moe1(torch::Tensor t_input, torch::Tensor t_output,
         int dh, int de, int nre, int nse, int ept, int n,
         torch::Tensor t_score, torch::Tensor t_top_indices,
         torch::Tensor t_token_count, torch::Tensor t_index_in_expert,
         torch::Tensor t_input_reorder) {

    __half *input_ptr   = (__half *)t_input.data_ptr();            // [n, dh]
    __half *output_ptr  = (__half *)t_output.data_ptr();           // [n, dh]

    __half *score = (__half *)t_score.data_ptr();                 // [n, nre]
    int *top_indices = (int *)t_top_indices.data_ptr();           // [n, ept]
    int *token_count = (int *)t_token_count.data_ptr();           // [nre]
    int *index_in_expert = (int *)t_index_in_expert.data_ptr();   // [n, ept]
    __half *input_reorder = (__half *)t_input_reorder.data_ptr(); // [n * ept, dh]

    // int *top_indices; // [n, ept], topk score
    if (ept == 4 && nre == 32) {
        top4_nre32<<<dim3((n+63)/64), dim3(64)>>>(score, top_indices, n);
    }
    else {
        topk<<<dim3((n+63)/64), dim3(64)>>>(score, top_indices, n, nre, ept);
    }

    // int *token_count; // [nre], token count of each expert
    // int *index_in_expert; // [n, ept], index in expert of each token, i -> expert token index
    // hipMemset(token_count, 0, nre * sizeof(int));
    for (int i = 0; i < nre; i++) {
        token_count[i] = 0;
    }
    if (nre == 32 && ept == 4) {
        calculate_index_nre32_ept4<<<dim3((n+63)/64), dim3(64)>>>(top_indices, token_count, index_in_expert, n);
    }
    else {
        calculate_index<<<dim3((n+63)/64), dim3(64)>>>(top_indices, token_count, index_in_expert, n, ept, nre);
    }

    // reorder input
    // __half *input_reorder; // [n * ept, dh]
    if (ept == 4 && nre == 32 && dh == 7168) {
        reorder_input_ept4nre32dh7168<<<dim3(n), dim3(64)>>>(input_ptr, top_indices, index_in_expert, input_reorder, n, ept, nre, dh);
    }
    else {
        reorder_input<<<dim3(n), dim3(64)>>>(input_ptr, top_indices, index_in_expert, input_reorder, n, ept, nre, dh);
    }

    hipDeviceSynchronize();
    int max_token_count = 0;
    for (int i = 0; i < nre; i++) {
        if (token_count[i] > max_token_count) max_token_count = token_count[i];
    }

    return max_token_count;
}

__global__ void pack_output(half *output_down_re, half *output_down_se, __half *output_ptr, const __half *score, int *top_indices, int *index_in_expert, int n, int ept, int nre, int nse, int dh, int max_count) {
    int idx = blockIdx.x;
    using half8 = __fp16 __attribute__((ext_vector_type(8)));
    if (idx < n) {
        for (int i = 0; i < ept; i++) {
            int expert_id = top_indices[idx * ept + i];
            int token_location = expert_id * max_count + index_in_expert[idx * ept + i];
            __half this_score = score[idx * nre + expert_id];
            __half zeros[8] = {__float2half(0.0f)};
            for (int j = threadIdx.x * 8; j < dh; j += 64 * 8) {
                if (i == 0) *(half8 *)&output_ptr[idx * dh + j] = *(half8 *)zeros;
                __half val[8], res[8];
                *(half8 *)val = *(half8 *)&output_down_re[token_location * dh + j];
                *(half8 *)res = *(half8 *)&output_ptr[idx * dh + j];
                for (int k = 0; k < 8; k++) {
                    res[k] = __hadd(__hmul(this_score, val[k]), res[k]);
                }
                *(half8 *)&output_ptr[idx * dh + j] = *(half8 *)res;
                // output_ptr[idx * dh + j] = __hadd(__hmul(this_score, output_down[token_location * dh + j]), output_ptr[idx * dh + j]);
            }
        }
        for (int i = 0; i < nse; i++) {
            for (int j = threadIdx.x * 8; j < dh; j += 64 * 8) {
                __half val[8], res[8];
                *(half8 *)val = *(half8 *)&output_down_se[(n * i + idx) * dh + j];
                *(half8 *)res = *(half8 *)&output_ptr[idx * dh + j];
                for (int k = 0; k < 8; k++) {
                    res[k] = __hadd(val[k], res[k]);
                }
                *(half8 *)&output_ptr[idx * dh + j] = *(half8 *)res;
                // output_ptr[idx * dh + j] = __hadd(output_down[(n * (nre + i) + idx) * dh + j], output_ptr[idx * dh + j]);
            }
        }
    }
}

__global__ void pack_output_ept4nre32nse1dh7168(half *output_down_re, half *output_down_se, __half *output_ptr, const __half *score, int *top_indices, int *index_in_expert, int n, int ept, int nre, int nse, int dh, int max_count) {
    int idx = blockIdx.x;
    using half8 = __fp16 __attribute__((ext_vector_type(8)));
    #pragma unroll
    for (int j = threadIdx.x * 8; j < 7168; j += 64 * 8) {
        __half res[8] = {__float2half(0.0f)};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int expert_id = top_indices[idx * 4 + i];
            int token_location = expert_id * max_count + index_in_expert[idx * 4 + i];
            __half this_score = score[idx * 32 + expert_id];
            __half val[8];
            *(half8 *)val = *(half8 *)&output_down_re[token_location * 7168 + j];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                res[k] = __hadd(__hmul(this_score, val[k]), res[k]);
            }
        }
        #pragma unroll
        for (int i = 0; i < 1; i++) {
            __half val[8];
            *(half8 *)val = *(half8 *)&output_down_se[(n * i + idx) * 7168 + j];
            for (int k = 0; k < 8; k++) {
                res[k] = __hadd(val[k], res[k]);
            }
        }
        *(half8 *)&output_ptr[idx * 7168 + j] = *(half8 *)res;
    }
}

void pack(torch::Tensor t_input, torch::Tensor t_output,
          int dh, int de, int nre, int nse, int ept, int n,
          torch::Tensor t_score, torch::Tensor t_top_indices,
          torch::Tensor t_token_count, torch::Tensor t_index_in_expert,
          torch::Tensor t_output_down_re, torch::Tensor t_output_down_se, int max_count) {

    __half *input_ptr   = (__half *)t_input.data_ptr();            // [n, dh]
    __half *output_ptr  = (__half *)t_output.data_ptr();           // [n, dh]

    __half *score = (__half *)t_score.data_ptr();                 // [n, nre]
    int *top_indices = (int *)t_top_indices.data_ptr();           // [n, ept]
    int *token_count = (int *)t_token_count.data_ptr();           // [nre]
    int *index_in_expert = (int *)t_index_in_expert.data_ptr();   // [n, ept]
    __half *output_down_re = (__half *)t_output_down_re.data_ptr();
    __half *output_down_se = (__half *)t_output_down_se.data_ptr();

    if (ept == 4 && nre == 32 && nse == 1 && dh == 7168) {
        pack_output_ept4nre32nse1dh7168<<<dim3(n), dim3(64)>>>(output_down_re, output_down_se, output_ptr, score, top_indices, index_in_expert, n, ept, nre, nse, dh, max_count);
    }
    else {
        pack_output<<<dim3(n), dim3(64)>>>(output_down_re, output_down_se, output_ptr, score, top_indices, index_in_expert, n, ept, nre, nse, dh, max_count);
    }
}

__global__ void se_silu_mul_kernel(__half *gate_ptr, const __half *up_ptr, int ne, int n, int d) {
    using float4 = float __attribute__((ext_vector_type(4)));
    int idx = blockIdx.y * n + blockIdx.x;
    __half gate_val[8], up_val[8], res[8];
    for (int i = threadIdx.x * 8; i < d; i += 256 * 8) {
        *(float4 *)&gate_val[0] = *(float4 *)&gate_ptr[idx * d + i];
        *(float4 *)&up_val[0] = *(float4 *)&up_ptr[idx * d + i];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float silu_val = __half2float(gate_val[j]) * (1.0f / (1.0f + expf(-__half2float(gate_val[j]))));
            res[j] = __hmul(__float2half(silu_val), up_val[j]);
        }
        *(float4 *)&gate_ptr[idx * d + i] = *(float4 *)&res[0];
    }
}

__global__ void se_silu_mul_kernel_d2048(__half *gate_ptr, const __half *up_ptr, int ne, int n, int d) {
    using float4 = float __attribute__((ext_vector_type(4)));
    int idx = blockIdx.y * n + blockIdx.x;
    __half gate_val[8], up_val[8], res[8];
    *(float4 *)&gate_val[0] = *(float4 *)&gate_ptr[idx * 2048 + threadIdx.x * 8];
    *(float4 *)&up_val[0] = *(float4 *)&up_ptr[idx * 2048 + threadIdx.x * 8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float silu_val = __half2float(gate_val[j]) * (1.0f / (1.0f + expf(-__half2float(gate_val[j]))));
        res[j] = __hmul(__float2half(silu_val), up_val[j]);
    }
    *(float4 *)&gate_ptr[idx * 2048 + threadIdx.x * 8] = *(float4 *)&res[0];
}

void se_silu_mul(torch::Tensor t_gate, torch::Tensor t_up, int ne, int n, int d) {
    __half *gate_ptr = (__half *)t_gate.data_ptr(); // [ne, n, d]
    __half *up_ptr   = (__half *)t_up.data_ptr();   // [ne, n, d]
    if (d == 2048) {
        se_silu_mul_kernel_d2048<<<dim3(n, ne), dim3(256)>>>(gate_ptr, up_ptr, ne, n, d);
    }
    else {
        se_silu_mul_kernel<<<dim3(n, ne), dim3(256)>>>(gate_ptr, up_ptr, ne, n, d);
    }
}

__global__ void re_silu_mul_kernel(__half *gate_ptr, const __half *up_ptr, int ne, int n, int d, int *token_count) {
    using float4 = float __attribute__((ext_vector_type(4)));
    if (blockIdx.x >= token_count[blockIdx.y]) return;
    int idx = blockIdx.y * n + blockIdx.x;
    __half gate_val[8], up_val[8], res[8];
    for (int i = threadIdx.x * 8; i < d; i += 64 * 8) {
        *(float4 *)&gate_val[0] = *(float4 *)&gate_ptr[idx * d + i];
        *(float4 *)&up_val[0] = *(float4 *)&up_ptr[idx * d + i];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float silu_val = __half2float(gate_val[j]) * (1.0f / (1.0f + expf(-__half2float(gate_val[j]))));
            res[j] = __hmul(__float2half(silu_val), up_val[j]);
        }
        *(float4 *)&gate_ptr[idx * d + i] = *(float4 *)&res[0];
    }
}

__global__ void re_silu_mul_kernel_d2048(__half *gate_ptr, const __half *up_ptr, int ne, int n, int d, int *token_count) {
    using float4 = float __attribute__((ext_vector_type(4)));
    if (blockIdx.x >= token_count[blockIdx.y]) return;
    int idx = blockIdx.y * n + blockIdx.x;
    __half gate_val[4][8], up_val[4][8], res[4][8];
    *(float4 *)&gate_val[0][0] = *(float4 *)&gate_ptr[idx * 2048 + threadIdx.x * 8];
    *(float4 *)&up_val[0][0] = *(float4 *)&up_ptr[idx * 2048 + threadIdx.x * 8];
    *(float4 *)&gate_val[1][0] = *(float4 *)&gate_ptr[idx * 2048 + threadIdx.x * 8 + 512];
    *(float4 *)&up_val[1][0] = *(float4 *)&up_ptr[idx * 2048 + threadIdx.x * 8 + 512];
    *(float4 *)&gate_val[2][0] = *(float4 *)&gate_ptr[idx * 2048 + threadIdx.x * 8 + 1024];
    *(float4 *)&up_val[2][0] = *(float4 *)&up_ptr[idx * 2048 + threadIdx.x * 8 + 1024];
    *(float4 *)&gate_val[3][0] = *(float4 *)&gate_ptr[idx * 2048 + threadIdx.x * 8 + 1536];
    *(float4 *)&up_val[3][0] = *(float4 *)&up_ptr[idx * 2048 + threadIdx.x * 8 + 1536];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float silu_val = __half2float(gate_val[i][j]) * (1.0f / (1.0f + expf(-__half2float(gate_val[i][j]))));
            res[i][j] = __hmul(__float2half(silu_val), up_val[i][j]);
        }
    }
    *(float4 *)&gate_ptr[idx * 2048 + threadIdx.x * 8] = *(float4 *)&res[0][0];
    *(float4 *)&gate_ptr[idx * 2048 + threadIdx.x * 8 + 512] = *(float4 *)&res[1][0];
    *(float4 *)&gate_ptr[idx * 2048 + threadIdx.x * 8 + 1024] = *(float4 *)&res[2][0];
    *(float4 *)&gate_ptr[idx * 2048 + threadIdx.x * 8 + 1536] = *(float4 *)&res[3][0];
}

void re_silu_mul(torch::Tensor t_gate, torch::Tensor t_up, int ne, int n, int d, torch::Tensor t_token_count) {
    __half *gate_ptr = (__half *)t_gate.data_ptr(); // [ne, n, d]
    __half *up_ptr   = (__half *)t_up.data_ptr();   // [ne, n, d]
    int *token_count = (int *)t_token_count.data_ptr(); // [ne]
    if (d == 2048) {
        re_silu_mul_kernel_d2048<<<dim3(n, ne), dim3(64)>>>(gate_ptr, up_ptr, ne, n, d, token_count);
    }
    else {
        re_silu_mul_kernel<<<dim3(n, ne), dim3(64)>>>(gate_ptr, up_ptr, ne, n, d, token_count);
    }
}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='moe',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['moe1', 'pack', 're_silu_mul', 'se_silu_mul'],
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

    gate_re = torch.empty(nre, dh, de, dtype=torch.float16, device='cuda')            # [nre, dh, de]
    up_re   = torch.empty(nre, dh, de, dtype=torch.float16, device='cuda')            # [nre, dh, de]
    down_re = torch.empty(nre, de, dh, dtype=torch.float16, device='cuda')            # [nre, de, dh]
    output = torch.empty(bs, sl, dh, dtype=torch.float16, device='cuda')              # [bs, sl, dh]

    top_indices = torch.empty(bs * sl, ept, dtype=torch.int32, device='cuda')         # [bs * sl, ept]
    token_count = torch.empty(nre, dtype=torch.int32, device='cuda')                  # [nre]
    index_in_expert = torch.empty(bs * sl, ept, dtype=torch.int32, device='cuda')     # [bs * sl, ept]
    input_reorder = torch.empty(nre, bs * sl, dh, dtype=torch.float16, device='cuda') # [nre, bs * sl, dh]

    input = input_tensor                                                              # [bs, sl, dh]
    router  = weights['router.weight']                                                # [nre, dh]
    for i in range(nre):
        gate_re[i] = weights[f'experts.{i}.0.weight']
        up_re[i] = weights[f'experts.{i}.1.weight']
        down_re[i] = weights[f'experts.{i}.2.weight']
    gate_se = weights['shared_experts.0.weight'].view(dh, nse, de).transpose(0, 1)    # [nse, dh, de]
    up_se   = weights['shared_experts.1.weight'].view(dh, nse, de).transpose(0, 1)    # [nse, dh, de]
    down_se = weights['shared_experts.2.weight'].view(nse, de, dh)                    # [nse, de, dh]

    score = torch.softmax(input.view(bs * sl, dh) @ router.T, dim=1)                  # [bs * sl, nre]

    max_count = module.moe1(input, output, dh, de, nre, nse, ept, bs * sl,
        score, top_indices, token_count, index_in_expert, input_reorder)

    hidden_re_gate = torch.bmm(input_reorder[:, :max_count, :], gate_re)              # [nre, max_count, de]
    hidden_re_up   = torch.bmm(input_reorder[:, :max_count, :], up_re)                # [nre, max_count, de]
    # hidden_re_gate = torch.nn.functional.silu(hidden_re_gate) * hidden_re_up        # [nre, max_count, de]
    module.re_silu_mul(hidden_re_gate, hidden_re_up, nre, max_count, de, token_count)
    hidden_se_gate = torch.matmul(input.view(bs * sl, dh), gate_se)                   # [nse, bs * sl, de]
    hidden_se_up   = torch.matmul(input.view(bs * sl, dh), up_se)                     # [nse, bs * sl, de]
    # hidden_se_gate = torch.nn.functional.silu(hidden_se_gate) * hidden_se_up        # [nse, bs * sl, de]
    module.se_silu_mul(hidden_se_gate, hidden_se_up, nse, bs * sl, de)
    output_down_re = torch.bmm(hidden_re_gate, down_re)                               # [nre, max_count, dh]
    output_down_se = torch.bmm(hidden_se_gate, down_se)                               # [nse, bs * sl, dh]

    module.pack(input, output, dh, de, nre, nse, ept, bs*sl,
        score, top_indices, token_count, index_in_expert, output_down_re, output_down_se, max_count)
    return output