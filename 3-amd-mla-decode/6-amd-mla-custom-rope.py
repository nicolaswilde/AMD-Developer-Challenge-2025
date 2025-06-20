import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CPP_WRAPPER = """
void mla_decode(torch::Tensor x, torch::Tensor y, int pos, int xoff, int yoff,
                int d0, int d1, int d2, int xs1, int xs2, int ys1, int ys2);
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

void mla_decode(torch::Tensor x, torch::Tensor y, int pos, int xoff, int yoff,
                int d0, int d1, int d2, int xs1, int xs2, int ys1, int ys2) {

    __hip_bfloat16 *x_ptr = (__hip_bfloat16 *)x.data_ptr() + xoff;
    __hip_bfloat16 *y_ptr = (__hip_bfloat16 *)y.data_ptr() + yoff;
    if (d0 % 2 == 0) {
        rope_3d_kernel_d0aligned2<<<dim3(d0/2, d1), dim3(64)>>>(d0, d1, d2, xs1, xs2, ys1, ys2, pos, x_ptr, y_ptr);
    }
    else {
        rope_3d_kernel<<<dim3(d0, d1), dim3(64)>>>(d0, d1, d2, xs1, xs2, ys1, ys2, pos, x_ptr, y_ptr);
    }

}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='mla_decode',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['mla_decode'],
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
    qk = torch.softmax(qk / ((dnope + drope) ** 0.5), dim=-1)                   # [bs, nh, pl+sl]
    # qk = torch.bmm(qCwUK, kv_cache.data[:, :pl+sl, :dkv].transpose(1, 2)) + torch.bmm(qU[:, :, dnope:], kR.transpose(1, 2))
    # qk = torch.softmax(qk / ((dnope + drope) ** 0.5), dim=-1)                   # [bs, nh, pl+sl]

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