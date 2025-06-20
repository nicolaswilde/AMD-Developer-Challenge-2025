import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CPP_WRAPPER = """
void mla_decode(torch::Tensor x, torch::Tensor y, int pos, int xoff, int yoff,
                int d0, int d1, int d2, int xs1, int xs2, int ys1, int ys2);
void div_softmax(torch::Tensor qk, int bs, int nh, int sl, int dnope, int drope);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

__global__ void rope_3d_kernel(int d0, int d1, int d2, int xs1, int xs2, int ys1, int ys2, int pos,
        const __hip_bfloat16 *x, __hip_bfloat16 *y) {
    int d0_idx = blockIdx.x;
    int d1_idx = blockIdx.y;
    __hip_bfloat16 idx = __hadd((__hip_bfloat16)pos, __hip_bfloat16(d1_idx));
    for (int d2_idx = threadIdx.x; d2_idx < d2/2; d2_idx += blockDim.x) {
        float x1 = (float)x[d0_idx * xs1 + d1_idx * xs2 + d2_idx];
        float x2 = (float)x[d0_idx * xs1 + d1_idx * xs2 + d2_idx + d2/2];
        __hip_bfloat16 theta = (__hip_bfloat16)d2_idx;
        theta = __hmul(theta, (__hip_bfloat16)(-2.0f));
        theta = __hdiv(theta, (__hip_bfloat16)d2);
        theta = __powf((__hip_bfloat16)10000.0f, theta);
        theta = __hmul(theta, (__hip_bfloat16)idx);
        __hip_bfloat16 cos = cosf((float)theta);
        __hip_bfloat16 sin = sinf((float)theta);
        y[d0_idx * ys1 + d1_idx * ys2 + d2_idx       ] = __hsub(__hmul(x1, cos), __hmul(x2, sin));
        y[d0_idx * ys1 + d1_idx * ys2 + d2_idx + d2/2] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
    }
}

__global__ void rope_3d_kernel_d0aligned2(int d0, int d1, int d2, int xs1, int xs2, int ys1, int ys2, int pos,
        const __hip_bfloat16 *x, __hip_bfloat16 *y) {
    int d0_idx = blockIdx.x * 2 + threadIdx.x / 32;
    int d1_idx = blockIdx.y;
    __hip_bfloat16 idx = __hadd((__hip_bfloat16)pos, __hip_bfloat16(d1_idx));
    for (int d2_idx = threadIdx.x % 32; d2_idx < d2/2; d2_idx += 32) {
        float x1 = (float)x[d0_idx * xs1 + d1_idx * xs2 + d2_idx];
        float x2 = (float)x[d0_idx * xs1 + d1_idx * xs2 + d2_idx + d2/2];
        __hip_bfloat16 theta = (__hip_bfloat16)d2_idx;
        theta = __hmul(theta, (__hip_bfloat16)(-2.0f));
        theta = __hdiv(theta, (__hip_bfloat16)d2);
        theta = __powf((__hip_bfloat16)10000.0f, theta);
        theta = __hmul(theta, (__hip_bfloat16)idx);
        __hip_bfloat16 cos = cosf((float)theta);
        __hip_bfloat16 sin = sinf((float)theta);
        y[d0_idx * ys1 + d1_idx * ys2 + d2_idx       ] = __hsub(__hmul(x1, cos), __hmul(x2, sin));
        y[d0_idx * ys1 + d1_idx * ys2 + d2_idx + d2/2] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
    }
}

__global__ void rope_3d_kernel_d0aligned16_d2is64(int d0, int d1, int d2, int xs1, int xs2, int ys1, int ys2, int pos,
        const __hip_bfloat16 *x, __hip_bfloat16 *y) {
    using float4 = float __attribute__((ext_vector_type(4)));
    int d0_idx = blockIdx.x * 16 + threadIdx.x / 4;
    int d1_idx = blockIdx.y;
    int d2_idx = (threadIdx.x % 4) * 8;
    __hip_bfloat16 x1[8], x2[8];
    *(float4 *)&x1[0] = *(float4 *)&x[d0_idx * xs1 + d1_idx * xs2 + d2_idx];
    *(float4 *)&x2[0] = *(float4 *)&x[d0_idx * xs1 + d1_idx * xs2 + d2_idx + d2/2];
    __hip_bfloat16 idx = __hadd((__hip_bfloat16)pos, __hip_bfloat16(d1_idx));
    __hip_bfloat16 y1[8], y2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        __hip_bfloat16 theta = (__hip_bfloat16)(d2_idx + i);
        theta = __hmul(theta, (__hip_bfloat16)(-2.0f));
        theta = __hdiv(theta, (__hip_bfloat16)d2);
        theta = __powf((__hip_bfloat16)10000.0f, theta);
        theta = __hmul(theta, (__hip_bfloat16)idx);
        __hip_bfloat16 cos = cosf((float)theta);
        __hip_bfloat16 sin = sinf((float)theta);
        y1[i] = __hsub(__hmul(x1[i], cos), __hmul(x2[i], sin));
        y2[i] = __hadd(__hmul(x2[i], cos), __hmul(x1[i], sin));
    }
    *(float4 *)&y[d0_idx * ys1 + d1_idx * ys2 + d2_idx       ] = *(float4 *)&y1[0];
    *(float4 *)&y[d0_idx * ys1 + d1_idx * ys2 + d2_idx + d2/2] = *(float4 *)&y2[0];
}

// rope_3d_kernel_d0is128_d2is64<<<dim3((d1+1)/2), dim3(64)>>>(d0, d1, d2, xs1, xs2, ys1, ys2, pos, x_ptr, y_ptr);
__global__ void rope_3d_kernel_d0is128_d2is64(int d0, int d1, int d2, int xs1, int xs2, int ys1, int ys2, int pos,
        const __hip_bfloat16 *x, __hip_bfloat16 *y) {

    int d1_idx = blockIdx.x * 2 + threadIdx.x / 32;
    if (d1_idx >= d1) return; // Prevent out-of-bounds access
    int d2_idx = (threadIdx.x % 16) * 2;
    __hip_bfloat16 idx = __hadd((__hip_bfloat16)pos, __hip_bfloat16(d1_idx));
    __hip_bfloat16 theta = (__hip_bfloat16)d2_idx;
    theta = __hmul(theta, (__hip_bfloat16)(-2.0f));
    theta = __hdiv(theta, (__hip_bfloat16)d2);
    theta = __powf((__hip_bfloat16)10000.0f, theta);
    theta = __hmul(theta, (__hip_bfloat16)idx);
    __hip_bfloat16 cos = cosf((float)theta);
    __hip_bfloat16 sin = sinf((float)theta);

    for (int d0_idx = (threadIdx.x % 32) / 16; d0_idx < 128; d0_idx += 2) {
        __hip_bfloat16 x1[2], x2[2], y1[2], y2[2];
        *(float *)&x1[0] = *(float *)&x[d0_idx * xs1 + d1_idx * xs2 + d2_idx];
        *(float *)&x2[0] = *(float *)&x[d0_idx * xs1 + d1_idx * xs2 + d2_idx + d2/2];
        y1[0] = __hsub(__hmul(x1[0], cos), __hmul(x2[0], sin));
        y1[1] = __hsub(__hmul(x1[1], cos), __hmul(x2[1], sin));
        y2[0] = __hadd(__hmul(x2[0], cos), __hmul(x1[0], sin));
        y2[1] = __hadd(__hmul(x2[1], cos), __hmul(x1[1], sin));
        *(float *)&y[d0_idx * ys1 + d1_idx * ys2 + d2_idx       ] = *(float *)&y1[0];
        *(float *)&y[d0_idx * ys1 + d1_idx * ys2 + d2_idx + d2/2] = *(float *)&y2[0];
    }
}

void mla_decode(torch::Tensor x, torch::Tensor y, int pos, int xoff, int yoff,
                int d0, int d1, int d2, int xs1, int xs2, int ys1, int ys2) {

    __hip_bfloat16 *x_ptr = (__hip_bfloat16 *)x.data_ptr() + xoff;
    __hip_bfloat16 *y_ptr = (__hip_bfloat16 *)y.data_ptr() + yoff;
    if (d0 == 128 && d2 == 64) {
        rope_3d_kernel_d0is128_d2is64<<<dim3((d1+1)/2), dim3(64)>>>(d0, d1, d2, xs1, xs2, ys1, ys2, pos, x_ptr, y_ptr);
    }
    // if (d2 == 64 && d0 % 16 == 0 && d0 * d1 > 16384) {
    //     // Use a specialized kernel for 64-dim and d1 multiple of 16
    //     rope_3d_kernel_d0aligned16_d2is64<<<dim3(d0/16, d1), dim3(64)>>>(d0, d1, d2, xs1, xs2, ys1, ys2, pos, x_ptr, y_ptr);
    // }
    else if (d0 % 2 == 0) {
        rope_3d_kernel_d0aligned2<<<dim3(d0/2, d1), dim3(64)>>>(d0, d1, d2, xs1, xs2, ys1, ys2, pos, x_ptr, y_ptr);
    }
    else {
        rope_3d_kernel<<<dim3(d0, d1), dim3(64)>>>(d0, d1, d2, xs1, xs2, ys1, ys2, pos, x_ptr, y_ptr);
    }
}

__global__ void div_softmax_kernel(__hip_bfloat16 *ptr, int bs, int nh, int sl, float scale) {
    int this_bs = blockIdx.x;
    int this_nh = blockIdx.y;
    int idx = this_bs * nh * sl + this_nh * sl;

    using float4 = float __attribute__((ext_vector_type(4)));

    __shared__ __hip_bfloat16 s_val[8192]; // msl = 8192
    __shared__ float sum_val[512];

    int sl_aligned = sl / 8 * 8;
    int sl_unaligned = sl % 8;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x * 8; i < sl_aligned; i += blockDim.x * 8) {
        __hip_bfloat16 val[8];
        float scale_val[8];
        *(float4 *)&val[0] = *(float4 *)&ptr[idx + i];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            scale_val[j] = (float)val[j] * scale; // Scale by 1/sqrt(dnope + drope)
            float exp_val = expf(scale_val[j]);
            thread_sum += exp_val;
            val[j] = (__hip_bfloat16)exp_val;
        }
        *(float4 *)&s_val[i] = *(float4 *)&val[0];
    }
    for (int i = sl_aligned + threadIdx.x; i < sl; i += blockDim.x) {
        __hip_bfloat16 val = ptr[idx + i];
        float scaled_val = (float)val * scale; // Scale by 1/sqrt(dnope + drope)
        float exp_val = expf(scaled_val);
        thread_sum += exp_val;
        s_val[i] = (__hip_bfloat16)exp_val;
    }
    sum_val[threadIdx.x] = thread_sum;

    __syncthreads();

    if (threadIdx.x < 64) {
        thread_sum = 0.0f;
        float this_val[8];
        *(float4 *)&this_val[0] = *(float4 *)&sum_val[threadIdx.x * 4];
        *(float4 *)&this_val[4] = *(float4 *)&sum_val[threadIdx.x * 4 + 256];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            thread_sum += this_val[i];
        }
        sum_val[threadIdx.x] = thread_sum;
    }

    if (threadIdx.x < 8) {
        thread_sum = 0.0f;
        float this_val[8];
        *(float4 *)&this_val[0] = *(float4 *)&sum_val[threadIdx.x * 4];
        *(float4 *)&this_val[4] = *(float4 *)&sum_val[threadIdx.x * 4 + 32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            thread_sum += this_val[i];
        }
        sum_val[threadIdx.x] = thread_sum;
    }

    if (threadIdx.x == 0) {
        thread_sum = 0.0f;
        float this_val[8];
        *(float4 *)&this_val[0] = *(float4 *)&sum_val[0];
        *(float4 *)&this_val[4] = *(float4 *)&sum_val[4];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            thread_sum += this_val[i];
        }
        sum_val[0] = thread_sum;
    }

    __syncthreads();

    float total_sum = sum_val[0];
    for (int i = threadIdx.x * 8; i < sl_aligned; i += blockDim.x * 8) {
        __hip_bfloat16 val[8];
        *(float4 *)&val[0] = *(float4 *)&s_val[i];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float softmax_val = (float)val[j] / total_sum;
            val[j] = (__hip_bfloat16)softmax_val;
        }
        *(float4 *)&ptr[idx + i] = *(float4 *)&val[0];
    }
    for (int i = sl_aligned + threadIdx.x; i < sl; i += blockDim.x) {
        __hip_bfloat16 val = s_val[i];
        float softmax_val = (float)val / total_sum;
        ptr[idx + i] = (__hip_bfloat16)softmax_val;
    }
}

void div_softmax(torch::Tensor qk, int bs, int nh, int sl, int dnope, int drope) {
    __hip_bfloat16 *qk_ptr = (__hip_bfloat16 *)qk.data_ptr();
    float scale = 1.0f / sqrtf((float)(dnope + drope));
    div_softmax_kernel<<<dim3(bs, nh), dim3(512)>>>(qk_ptr, bs, nh, sl, scale);
}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='mla_decode',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['mla_decode', 'div_softmax'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O3", "-w"],
)

def custom_kernel(data: input_t) -> output_t:
    config, x, kv_cache = data

    bs = config.batch_size
    sl = config.seq_len
    pl = kv_cache.seq_len
    msl = config.max_seq_len
    nh = config.n_heads
    d =  config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    wDQ  = config.Q_proj_down_weight                                            # [dq, d]
    wDKV = config.KV_proj_down_weight                                           # [dkv + drope, d]
    wUQ  = config.Q_proj_up_weight                                              # [nh * (dnope + drope), dq]
    wUKV = config.KV_proj_up_weight                                             # [nh * (dnope + dv), dkv]
    wO   = config.wo_weight                                                     # [d, nh * dv]

    x = x.view(bs, d)                                                           # [bs, d]
    qD = x @ wDQ.T                                                              # [bs, dq] = [bs, d] @ [dq, d]
    qU = (qD @ wUQ.T)                                                           # [bs, nh * (dnope + drope)] = [bs, dq] @ [nh * (dnope + drope), dq]
    qU = qU.view(bs * nh, sl, dnope + drope)                                    # [bs * nh, sl, dnope + drope]
    # qU[:, :, dnope:] = rope(qU[:, :, dnope:], pl)
    module.mla_decode(qU, qU, pl, dnope, dnope, bs * nh, sl, drope, dnope + drope, dnope + drope, dnope + drope, dnope + drope)
    qU = qU.view(bs, nh, dnope + drope)                                         # [bs, nh, dnope + drope]

    kv_cache.data[:, pl:pl+sl, :] = (x @ wDKV.T).view(bs, sl, dkv + drope)      # [bs, sl, dkv + drope] = [bs, d] @ [dkv + drope, d]
    kv_cache.seq_len += 1
    # kR = rope(kv_cache.data[:, :pl+sl, dkv:], 0).view(bs, pl+sl, drope)
    kR = torch.empty((bs, pl + sl, drope), dtype=torch.bfloat16, device='cuda')
    module.mla_decode(kv_cache.data, kR, 0, dkv, 0, bs, pl + sl, drope, msl * (dkv + drope), dkv + drope, (pl + sl) * drope, drope)

    wUKV = wUKV.view(nh, dnope + dv, dkv)                                       # [nh, dnope + dv, dkv]
    # qCwUK = torch.empty((bs, nh, dkv), dtype=torch.bfloat16, device='cuda')
    # for i_nh in range(nh):
    #     qCwUK[:, i_nh, :] = qU[:, i_nh, :dnope] @ wUKV[i_nh, :dnope, :]
    qCwUK = torch.bmm(qU[:, :, :dnope].transpose(0, 1), wUKV[:, :dnope, :])     # [nh, bs, dkv] = [bs, nh, dnope] @ [nh, dnope, dkv]

    # qk = torch.empty((bs, nh, pl+sl), dtype=torch.bfloat16, device='cuda')
    # for i_bs in range(bs):
    #     qk[i_bs, :, :] = qCwUK[i_bs, :, :] @ kv_cache.data[i_bs, :pl+sl, :dkv].T + qU[i_bs, :, dnope:] @ kR[i_bs, :pl+sl, :].T
    qk = torch.bmm(qCwUK.transpose(0, 1), kv_cache.data[:, :pl+sl, :dkv].transpose(1, 2)) # [bs, nh, pl+sl] = [nh, bs, dkv] @ [bs, pl+sl, dkv]
    qk += torch.bmm(qU[:, :, dnope:], kR.transpose(1, 2))                       # [bs, nh, pl+sl] = [bs, nh, drope] @ [bs, pl+sl, drope]
    # qk = torch.bmm(qCwUK, kv_cache.data[:, :pl+sl, :dkv].transpose(1, 2)) + torch.bmm(qU[:, :, dnope:], kR.transpose(1, 2))

    # qk = torch.softmax(qk / ((dnope + drope) ** 0.5), dim=-1)                 # [bs, nh, pl+sl]
    module.div_softmax(qk, bs, nh, pl + sl, dnope, drope) # Custom softmax kernel


    # qkkvC = torch.empty((bs, nh, dkv), dtype=torch.bfloat16, device='cuda')
    # for i_bs in range(bs):
    #     qkkvC[i_bs, :, :] = qk[i_bs, :, :] @ kv_cache.data[i_bs, :pl+sl, :dkv]
    qkkvC = torch.bmm(qk, kv_cache.data[:, :pl+sl, :dkv])                       # [bs, nh, dkv] = [bs, nh, pl+sl] @ [bs, pl+sl, dkv]

    # qkv = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device='cuda')
    # for i_nh in range(nh):
    #     qkv[:, i_nh, :] = qkkvC[:, i_nh, :] @ wUKV[i_nh, dnope:, :].T
    qkv = torch.bmm(qkkvC.transpose(0, 1), wUKV[:, dnope:, :].transpose(1, 2)).transpose(0, 1) # [bs, nh, dv] = [bs, nh, dkv] @ [nh, dv, dkv]

    output = (qkv.reshape(bs, nh * dv) @ wO.T).view(bs, sl, d)                  # [bs, sl, d] = [bs, nh * dv] @ [d, nh * dv]

    return output, kv_cache.data