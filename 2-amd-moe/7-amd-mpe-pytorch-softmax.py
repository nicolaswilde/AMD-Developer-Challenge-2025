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
         torch::Tensor t_expert_offset, torch::Tensor t_input_reorder,
         torch::Tensor t_hidden, torch::Tensor t_output_down);
int moe2(torch::Tensor t_input, torch::Tensor t_output,
         int dh, int de, int nre, int nse, int ept, int n,
         torch::Tensor t_score, torch::Tensor t_top_indices,
         torch::Tensor t_token_count, torch::Tensor t_index_in_expert,
         torch::Tensor t_expert_offset, torch::Tensor t_input_reorder,
         torch::Tensor t_hidden, torch::Tensor t_output_down);
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

__global__ void pack_output(half *output_down, __half *output_ptr, const __half *score, int *top_indices, int *expert_offset, int *index_in_expert, int n, int ept, int nre, int nse, int dh) {
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
                result += __half2float(output_down[(n * (nre + i) + idx) * dh + j]);
            }
            output_ptr[idx * dh + j] = __float2half(result);
        }
    }
}

int moe1(torch::Tensor t_input, torch::Tensor t_output,
         int dh, int de, int nre, int nse, int ept, int n,
         torch::Tensor t_score, torch::Tensor t_top_indices,
         torch::Tensor t_token_count, torch::Tensor t_index_in_expert,
         torch::Tensor t_expert_offset, torch::Tensor t_input_reorder,
         torch::Tensor t_hidden, torch::Tensor t_output_down) {

    __half *input_ptr   = (__half *)t_input.data_ptr();            // [n, dh]
    __half *output_ptr  = (__half *)t_output.data_ptr();           // [n, dh]

    __half *score = (__half *)t_score.data_ptr();                 // [n, nre]
    int *top_indices = (int *)t_top_indices.data_ptr();           // [n, ept]
    int *token_count = (int *)t_token_count.data_ptr();           // [nre]
    int *index_in_expert = (int *)t_index_in_expert.data_ptr();   // [n, ept]
    int *expert_offset = (int *)t_expert_offset.data_ptr();       // [nre]
    __half *input_reorder = (__half *)t_input_reorder.data_ptr(); // [n * ept, dh]
    __half *hidden = (__half *)t_hidden.data_ptr();               // [n * (ept + nse), de]
    __half *output_down = (__half *)t_output_down.data_ptr();     // [n * (ept + nse), dh]

    // int *top_indices; // [n, ept], topk score
    topk<<<dim3((n+63)/64), dim3(64)>>>(score, top_indices, n, nre, ept);

    // int *token_count; // [nre], token count of each expert
    // int *index_in_expert; // [n, ept], index in expert of each token, i -> expert token index
    hipMemset(token_count, 0, nre * sizeof(int));
    calculate_index<<<dim3((n+63)/64), dim3(64)>>>(top_indices, token_count, index_in_expert, n, ept, nre);
    hipDeviceSynchronize();

    // expert offset
    // int *expert_offset; // [nre]
    {
        int offset = 0;
        for (int i = 0; i < nre; i++) {
            expert_offset[i] = i * n;
        }
    }

    // reorder input
    // __half *input_reorder; // [n * ept, dh]
    reorder_input<<<dim3(n), dim3(64)>>>(input_ptr, top_indices, expert_offset, index_in_expert, input_reorder, n, ept, nre, dh);

    int max_token_count = 0;
    for (int i = 0; i < nre; i++) {
        if (token_count[i] > max_token_count) max_token_count = token_count[i];
    }

    return max_token_count;
}

void moe2(torch::Tensor t_input, torch::Tensor t_output,
         int dh, int de, int nre, int nse, int ept, int n,
         torch::Tensor t_score, torch::Tensor t_top_indices,
         torch::Tensor t_token_count, torch::Tensor t_index_in_expert,
         torch::Tensor t_expert_offset, torch::Tensor t_input_reorder,
         torch::Tensor t_hidden, torch::Tensor t_output_down) {

    __half *input_ptr   = (__half *)t_input.data_ptr();            // [n, dh]
    __half *output_ptr  = (__half *)t_output.data_ptr();           // [n, dh]

    __half *score = (__half *)t_score.data_ptr();                 // [n, nre]
    int *top_indices = (int *)t_top_indices.data_ptr();           // [n, ept]
    int *token_count = (int *)t_token_count.data_ptr();           // [nre]
    int *index_in_expert = (int *)t_index_in_expert.data_ptr();   // [n, ept]
    int *expert_offset = (int *)t_expert_offset.data_ptr();       // [nre]
    __half *input_reorder = (__half *)t_input_reorder.data_ptr(); // [n * ept, dh]
    __half *hidden = (__half *)t_hidden.data_ptr();               // [n * (ept + nse), de]
    __half *output_down = (__half *)t_output_down.data_ptr();     // [n * (ept + nse), dh]

    pack_output<<<dim3(n), dim3(64)>>>(output_down, output_ptr, score, top_indices, expert_offset, index_in_expert, n, ept, nre, nse, dh);
}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='moe',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['moe1', 'moe2'],
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
    gate   = torch.empty(nre + nse, dh, de, dtype=torch.float16, device='cuda') # [nre + nse, dh, de]
    up     = torch.empty(nre + nse, dh, de, dtype=torch.float16, device='cuda') # [nre + nse, dh, de]
    down   = torch.empty(nre + nse, de, dh, dtype=torch.float16, device='cuda') # [nre + nse, de, dh]
    output = torch.zeros(bs, sl, dh, dtype=torch.float16, device='cuda')        # [bs, sl, dh]
    for i in range(nre):
        gate[i] = weights[f'experts.{i}.0.weight']
        up[i] = weights[f'experts.{i}.1.weight']
        down[i] = weights[f'experts.{i}.2.weight']
    for i in range(nse):
        gate[nre + i] = weights['shared_experts.0.weight'][:, i * de:(i + 1) * de]
        up[nre + i] = weights['shared_experts.1.weight'][:, i * de:(i + 1) * de]
        down[nre + i] = weights['shared_experts.2.weight'][i * de:(i + 1) * de, :]

    top_indices = torch.empty(bs * sl * ept, dtype=torch.int32, device='cuda')                 # [bs * sl, ept]
    token_count = torch.empty(nre, dtype=torch.int32, device='cuda')                           # [nre]
    index_in_expert = torch.empty(bs * sl * ept, dtype=torch.int32, device='cuda')             # [bs * sl, ept]
    expert_offset = torch.empty(nre, dtype=torch.int32, device='cuda')                         # [nre]
    input_reorder = torch.empty(nre, bs*sl, dh, dtype=torch.float16, device='cuda')            # [nre, bs * sl, dh]
    hidden = torch.empty(bs * sl * (nre + nse) * de, dtype=torch.float16, device='cuda')       # [bs * sl * (nre + nse), de]
    output_down = torch.empty(bs * sl * (nre + nse) * dh, dtype=torch.float16, device='cuda')  # [bs * sl * (nre + nse), dh]

    score = torch.softmax(input.view(bs*sl, dh) @ router.T, dim=1)  # [bs * sl, nre]

    max_count = module.moe1(input, output, dh, de, nre, nse, ept, bs*sl,
        score, top_indices, token_count, index_in_expert, expert_offset,
        input_reorder, hidden, output_down)

    hidden = torch.empty(nre + nse, bs * sl, de, dtype=torch.float16, device='cuda')  # [nre + nse, bs * sl, de]
    hidden[:nre, :max_count, :] = torch.nn.functional.silu(torch.bmm(input_reorder[:, :max_count, :], gate[:nre, :, :])) * torch.bmm(input_reorder[:, :max_count, :], up[:nre, :, :])
    hidden[nre:, :, :] = torch.nn.functional.silu(torch.matmul(input.view(bs*sl, dh), gate[nre:, :, :])) * torch.matmul(input.view(bs*sl, dh), up[nre:, :, :])
    output_down = torch.empty(nre + nse, bs * sl, dh, dtype=torch.float16, device='cuda')  # [nre + nse, bs * sl, dh]
    output_down[:nre, :max_count, :] = torch.bmm(hidden[:nre, :max_count, :], down[:nre, :, :])
    output_down[nre:, :, :] = torch.bmm(hidden[nre:, :, :], down[nre:, :, :])

    module.moe2(input, output, dh, de, nre, nse, ept, bs*sl,
        score, top_indices, token_count, index_in_expert, expert_offset,
        input_reorder, hidden, output_down)
    return output