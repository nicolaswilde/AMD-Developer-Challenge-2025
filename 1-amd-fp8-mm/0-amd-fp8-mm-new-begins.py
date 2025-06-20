import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CPP_WRAPPER = """
void dqa(torch::Tensor dst, torch::Tensor src, torch::Tensor as, int m, int k);
void dqb(torch::Tensor dst, torch::Tensor src, torch::Tensor bs, int n, int k);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>

__global__ void dqa_kernel(float* dst, __hip_fp8_e4m3_fnuz* src, float* as, int m, int k) {
    int k_idx = blockIdx.x;
    for (int m_idx = threadIdx.x; m_idx < m; m_idx += blockDim.x) {
        __hip_fp8_e4m3_fnuz src_val = src[k_idx * m + m_idx];
        float a_scale = as[k_idx / 128 * m + m_idx];
        float dst_val = (float)src_val * a_scale;
        dst[k_idx * m + m_idx] = dst_val;
    }
}

void dqa(torch::Tensor dst, torch::Tensor src, torch::Tensor as, int m, int k) {
    float* dst_ptr = (float*)dst.data_ptr();                             // [k, m]
    __hip_fp8_e4m3_fnuz* src_ptr = (__hip_fp8_e4m3_fnuz*)src.data_ptr(); // [k, m]
    float* as_ptr = (float*)as.data_ptr();                               // [k // 128, m]
    dqa_kernel<<<dim3(k), dim3(64)>>>(dst_ptr, src_ptr, as_ptr, m, k);
}

__global__ void dqb_kernel(float* dst, __hip_fp8_e4m3_fnuz* src, float* bs, int n, int k) {
    int k_idx = blockIdx.x;
    for (int n_idx = threadIdx.x; n_idx < n; n_idx += blockDim.x) {
        __hip_fp8_e4m3_fnuz src_val = src[k_idx * n + n_idx];
        float b_scale = bs[k_idx / 128 * ((n + 127) / 128) + n_idx / 128];
        float dst_val = (float)src_val * b_scale;
        dst[k_idx * n + n_idx] = dst_val;
    }
}

void dqb(torch::Tensor dst, torch::Tensor src, torch::Tensor bs, int n, int k) {
    float* dst_ptr = (float*)dst.data_ptr();                             // [k, n]
    __hip_fp8_e4m3_fnuz* src_ptr = (__hip_fp8_e4m3_fnuz*)src.data_ptr(); // [k, n]
    float* bs_ptr = (float*)bs.data_ptr();                               // [k // 128, n // 128]
    dqb_kernel<<<dim3(k), dim3(64)>>>(dst_ptr, src_ptr, bs_ptr, n, k);
}

"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='fp8_mm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['dqa', 'dqb'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O3"],
)

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemm
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k],
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k],
            a_scale: torch.Tensor[float32] of shape [m, k // 128],
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128],
            c: torch.Tensor[bfloat16] of shape [m, n]
    Returns:
        Tensor containing output in bf16
    """
    # c: [m, n] is pre-allocated memory to avoid timing allocation overhead.
    a, b, a_scale, b_scale, c = data

    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]

    a_fp32 = torch.empty(k, m, dtype=torch.float32, device='cuda')
    b_fp32 = torch.empty(k, n, dtype=torch.float32, device='cuda')
    module.dqa(a_fp32, a, a_scale, m, k)
    module.dqb(b_fp32, b, b_scale, n, k)
    return a_fp32.T @ b_fp32
