# This script provides a template for using load_inline to run a HIP kernel for
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
CPP_WRAPPER = """
void moe(torch::Tensor input, torch::Tensor router,
         torch::Tensor gate, torch::Tensor up, torch::Tensor down,
         torch::Tensor output,
         int dh, int de, int nre, int nse, int ept, int n);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/amd_detail/amd_hip_fp16.h>

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

__global__ void topk_fp32(const float *input, int *top_indices, int m, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        // Find top-k indices for row idx
        // Simple selection sort for small k
        for (int i = 0; i < k; ++i) {
            float max_val = -INFINITY;
            int max_idx = -1;
            for (int j = 0; j < n; ++j) {
                if (input[idx * n + j] > max_val) {
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

__global__ void topk_fp16(const __half *input, int *top_indices, int m, int n, int k) {
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

__global__ void calculate_index(int *top_indices, int *token_count, int *token_index, int *index_in_expert, int n, int ept, int nre) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nre) {
        int count = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < ept; ++j) {
                if (top_indices[i * ept + j] == idx) {
                    token_index[idx * n + count] = i;
                    index_in_expert[i * ept + j] = count;
                    count++;
                }
            }
        }
        token_count[idx] = count;
    }
}

__global__ void calculate_offset(int *token_count, int *expert_offset, int nre) {
    int offset = 0;
    for (int i = 0; i < nre; ++i) {
        expert_offset[i] = offset;
        offset += token_count[i];
    }
}

__global__ void reorder_input(__half *input, int *top_indices, int *expert_offset, int *index_in_expert, __half *input_reorder, int n, int ept, int nre, int dh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < ept; ++i) {
            int expert_id = top_indices[idx * ept + i];
            int token_location = expert_offset[expert_id] + index_in_expert[idx * ept + i];
            for (int j = 0; j < dh; ++j) {
                input_reorder[token_location * dh + j] = input[idx * dh + j];
            }
        }
    }
}

__global__ void silu_gate_mul_up(float *gate, float *up, __half *hidden, int n, int de) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < de; ++i) {
            float gate_val = gate[idx * de + i];
            float silu_val = gate_val * (1.0f / (1.0f + __expf(-gate_val)));
            hidden[idx * de + i] = __float2half(silu_val * up[idx * de + i]);
        }
    }
}

__global__ void pack_output(float *output_down, __half *output_ptr, const __half *score, int *top_indices, int *token_count, int *expert_offset, int *index_in_expert, int n, int ept, int nre, int nse, int dh) {
    int idx = blockIdx.x;
    if (idx < n) {
        for (int j = threadIdx.x; j < dh; j += 64) {
            float result = 0.0f;
            for (int i = 0; i < ept; ++i) {
                int expert_id = top_indices[idx * ept + i];
                int token_location = expert_offset[expert_id] + index_in_expert[idx * ept + i];
                result += __half2float(score[idx * nre + expert_id]) * output_down[token_location * dh + j];
            }
            for (int i = 0; i < nse; ++i) {
                result += output_down[(n * (ept + i) + idx) * dh + j];
            }
            output_ptr[idx * dh + j] = __float2half(result);
        }
    }
}

void hgemm_mk_kn_mn_fp16acc(hipblasHandle_t handle,
    __half *a, __half *b, __half *c, int m, int n, int k, __half *alpha, __half *beta) {

    hipblasHgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
        (const hipblasHalf *)alpha, (const hipblasHalf *)b, n, (const hipblasHalf *)a, k,
        (const hipblasHalf *)beta, (hipblasHalf *)c, n);
}

void hgemm_mk_nk_mn_fp16acc(hipblasHandle_t handle,
    __half *a, __half *b, __half *c, int m, int n, int k, __half *alpha, __half *beta) {

    hipblasHgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, k,
        (const hipblasHalf *)alpha, (const hipblasHalf *)b, k, (const hipblasHalf *)a, k,
        (const hipblasHalf *)beta, (hipblasHalf *)c, n);
}

void hgemm_mk_kn_mn_fp32acc(hipblasHandle_t handle,
    __half *a, __half *b, float *c, int m, int n, int k, float *alpha, float *beta) {

    hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
        alpha, (const hipblasHalf *)b, HIP_R_16F, n, (const hipblasHalf *)a, HIP_R_16F, k,
        beta, c, HIP_R_32F, n, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
}

void hgemm_mk_nk_mn_fp32acc(hipblasHandle_t handle,
    __half *a, __half *b, float *c, int m, int n, int k, float *alpha, float *beta) {

    hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, k,
        alpha, (const hipblasHalf *)b, HIP_R_16F, k, (const hipblasHalf *)a, HIP_R_16F, k,
        beta, c, HIP_R_32F, n, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
}

void sgemm_mk_kn_mn_fp32acc(hipblasHandle_t handle,
    float *a, float *b, float *c, int m, int n, int k, float *alpha, float *beta) {

    hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
        alpha, (const float *)b, n, (const float *)a, k,
        beta, c, n);
}

void sgemm_mk_nk_mn_fp32acc(hipblasHandle_t handle,
    float *a, float *b, float *c, int m, int n, int k, float *alpha, float *beta) {

    hipblasSgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, k,
        alpha, (const float *)b, n, (const float *)a, k,
        beta, c, n);
}

void moe(torch::Tensor input, torch::Tensor router,
         torch::Tensor gate, torch::Tensor up, torch::Tensor down,
         torch::Tensor output,
         int dh, int de, int nre, int nse, int ept, int n) {

hipEvent_t start, end;
hipEventCreate(&start);
hipEventCreate(&end);
hipEventRecord(start);

        __half *input_ptr = (__half *)input.data_ptr();   // [n, dh]
        __half *router_ptr = (__half *)router.data_ptr(); // [nre, dh]
        __half *gate_ptr = (__half *)gate.data_ptr();     // [nre + nse, dh, de]
        __half *up_ptr = (__half *)up.data_ptr();         // [nre + nse, dh, de]
        __half *down_ptr = (__half *)down.data_ptr();     // [nre + nse, de, dh]
        __half *output_ptr = (__half *)output.data_ptr(); // [n, dh]

        hipblasHandle_t handle;
        hipblasCreate(&handle);
        float alpha_fp32 = 1.0f, beta_fp32 = 0.0f;
        __half alpha_fp16 = __float2half(1.0f), beta_fp16 = __float2half(0.0f);
        // hipblasHgemm:
        // 1. a [m, k], b[k, n], c[m, n]
        //    HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k, alpha, b, n, a, k, beta, c, n
        // 2. a [m, k], b[n, k], c[m, n]
        //    HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, k, alpha, b, k, a, k, beta, c, n

        // expert score
        float *score_fp32; // [n, nre]
        hipMalloc((void**)&score_fp32, n * nre * sizeof(float));
        // input_ptr [n, dh] @ router_ptr [nre, dh] -> score [n, nre], M = n, N = nre, K = dh
        hgemm_mk_nk_mn_fp32acc(handle, input_ptr, router_ptr, score_fp32, n, nre, dh, &alpha_fp32, &beta_fp32);
        __half *score_fp16; // [n, nre]
        hipMalloc((void**)&score_fp16, n * nre * sizeof(__half));
        softmax_fp32_fp16<<<dim3((n+63)/64), dim3(64)>>>(score_fp32, score_fp16, n, nre);

        // topk score
        int *top_indices; // [n, ept]
        hipMalloc((void**)&top_indices, n * ept * sizeof(int));
        topk_fp16<<<dim3((n+63)/64), dim3(64)>>>(score_fp16, top_indices, n, nre, ept);

        // token count of each expert
        int *token_count; // [nre]
        hipMalloc((void**)&token_count, nre * sizeof(int));

        // token index of each expert, i -> token_id
        int *token_index; // [nre, n]
        hipMalloc((void**)&token_index, nre * n * sizeof(int));

        // index in expert of each token, i -> expert token index
        int *index_in_expert; // [n, ept]
        hipMalloc((void**)&index_in_expert, n * ept * sizeof(int));
        calculate_index<<<dim3((nre+63)/64), dim3(64)>>>(top_indices, token_count, token_index, index_in_expert, n, ept, nre);
        hipDeviceSynchronize();

        // expert offset
        int *expert_offset; // [nre]
        hipMalloc((void**)&expert_offset, nre * sizeof(int));
        calculate_offset<<<dim3(1), dim3(1)>>>(token_count, expert_offset, nre);

        // reorder input
        __half *input_reorder; // [n * ept, dh]
        hipMalloc((void**)&input_reorder, n * ept * dh * sizeof(__half));
        reorder_input<<<dim3((n+63)/64), dim3(64)>>>(input_ptr, top_indices, expert_offset, index_in_expert, input_reorder, n, ept, nre, dh);

        // gate output
        float *output_gate; // [n * (ept + nse), de]
        hipMalloc((void**)&output_gate, n * (ept + nse) * de * sizeof(float));
        for (int i = 0; i < nre; ++i) {
            hgemm_mk_kn_mn_fp32acc(handle,
                input_reorder + expert_offset[i] * dh,
                gate_ptr + i * dh * de,
                output_gate + expert_offset[i] * de,
                token_count[i], de, dh, &alpha_fp32, &beta_fp32);
        }
        for (int i = 0; i < nse; ++i) {
            hgemm_mk_kn_mn_fp32acc(handle,
                input_ptr,
                gate_ptr + (nre + i) * dh * de,
                output_gate + (n * (ept + i)) * de,
                n, de, dh, &alpha_fp32, &beta_fp32);
        }

        // up output
        float *output_up; // [n * (ept + nse), de]
        hipMalloc((void**)&output_up, n * (ept + nse) * de * sizeof(float));
        for (int i = 0; i < nre; ++i) {
            hgemm_mk_kn_mn_fp32acc(handle,
                input_reorder + expert_offset[i] * dh,
                up_ptr + i * dh * de,
                output_up + expert_offset[i] * de,
                token_count[i], de, dh, &alpha_fp32, &beta_fp32);
        }
        for (int i = 0; i < nse; ++i) {
            hgemm_mk_kn_mn_fp32acc(handle,
                input_ptr,
                up_ptr + (nre + i) * dh * de,
                output_up + (n * (ept + i)) * de,
                n, de, dh, &alpha_fp32, &beta_fp32);
        }

        // hidden state: silu(output_gate) * output_up
        __half *hidden; // [n * (ept + nse), de]
        hipMalloc((void**)&hidden, n * (ept + nse) * de * sizeof(__half));
        silu_gate_mul_up<<<dim3((n * (ept + nse) + 63) / 64), dim3(64)>>>(output_gate, output_up, hidden, n * (ept + nse), de);

        // down output
        float *output_down; // [n * (ept + nse), dh]
        hipMalloc((void**)&output_down, n * (ept + nse) * dh * sizeof(float));
        //    HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, k, alpha, b, k, a, k, beta, c, n
        for (int i = 0; i < nre; ++i) {
            hgemm_mk_kn_mn_fp32acc(handle,
                hidden + expert_offset[i] * de,
                down_ptr + i * de * dh,
                output_down + expert_offset[i] * dh,
                token_count[i], dh, de, &alpha_fp32, &beta_fp32);
        }
        for (int i = 0; i < nse; ++i) {
            hgemm_mk_kn_mn_fp32acc(handle,
                hidden + (n * (ept + i)) * de,
                down_ptr + (nre + i) * de * dh,
                output_down + (n * (ept + i)) * dh,
                n, dh, de, &alpha_fp32, &beta_fp32);
        }

        // pack output
        pack_output<<<dim3(n), dim3(64)>>>(output_down, output_ptr, score_fp16, top_indices, token_count, expert_offset, index_in_expert, n, ept, nre, nse, dh);

        // free memory
        hipFree(score_fp32);
        hipFree(score_fp16);
        hipFree(top_indices);
        hipFree(token_count);
        hipFree(token_index);
        hipFree(index_in_expert);
        hipFree(expert_offset);
        hipFree(input_reorder);
        hipFree(output_gate);
        hipFree(output_up);
        hipFree(hidden);
        hipFree(output_down);
        hipblasDestroy(handle);

hipEventRecord(end);
hipEventSynchronize(end);
float msec;
hipEventElapsedTime(&msec, start, end);
// output_ptr[0] = __float2half(msec);
hipEventDestroy(start);
hipEventDestroy(end);

        hipDeviceSynchronize();
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

    input = input_tensor                                                        # [bs, sl, dh]
    router = weights['router.weight']                                           # [nre, dh]
    gate = torch.empty(nre + nse, dh, de, dtype=torch.float16, device='cuda')   # [nre + nse, dh, de]
    up = torch.empty(nre + nse, dh, de, dtype=torch.float16, device='cuda')     # [nre + nse, dh, de]
    down = torch.empty(nre + nse, de, dh, dtype=torch.float16, device='cuda')   # [nre + nse, de, dh]
    for i in range(nre):
        gate[i] = weights[f'experts.{i}.0.weight']
        up[i] = weights[f'experts.{i}.1.weight']
        down[i] = weights[f'experts.{i}.2.weight']
    for i in range(nse):
        gate[nre + i] = weights['shared_experts.0.weight'][:, i * de:(i + 1) * de]
        up[nre + i] = weights['shared_experts.1.weight'][:, i * de:(i + 1) * de]
        down[nre + i] = weights['shared_experts.2.weight'][i * de:(i + 1) * de, :]
    output = torch.zeros(bs, sl, dh, dtype=torch.float16, device='cuda')        # [bs, sl, dh]
    module.moe(input, router, gate, up, down, output, dh, de, nre, nse, ept, bs*sl)
    # with open("amd-fp8-moe-hip-naive.log", "a") as f:
    #     f.write(f"bs {bs}, sl {sl}, dh {dh}, de {de}, nre {nre}, nse {nse}, ept {ept}, Time: {output[0][0][0].to(torch.float32) * 1000} ms\n")
    return output