import math
import torch
from torch import nn
from task import input_t, output_t

def rope(x: torch.Tensor, pos: int) -> torch.Tensor:

    sl = x.shape[0]
    d = x.shape[-1]

    theta = torch.arange(d // 2, dtype=torch.bfloat16, device='cuda')
    theta = 10000 ** (-2 * theta / d)
    idx = torch.arange(pos, pos + sl, dtype=torch.bfloat16, device='cuda')
    idx_theta = torch.einsum('s,d->sd', idx, theta)
    cos = torch.cos(idx_theta).to(torch.bfloat16)
    sin = torch.sin(idx_theta).to(torch.bfloat16)

    if x.dim() == 2:
        y = torch.zeros_like(x)
        x1 = x[:, :d//2]
        x2 = x[:, d//2:]
        y[:, :d//2] += x1 * cos
        y[:, d//2:] += x2 * cos
        y[:, :d//2] += -x2 * sin
        y[:, d//2:] += x1 * sin
        return y
    elif x.dim() == 3:
        nh = x.shape[1]
        y = torch.zeros_like(x)
        for i_nh in range(nh):
            x1 = x[:, i_nh, :d//2]
            x2 = x[:, i_nh, d//2:]
            y[:, i_nh, :d//2] += x1 * cos
            y[:, i_nh, d//2:] += x2 * cos
            y[:, i_nh, :d//2] += -x2 * sin
            y[:, i_nh, d//2:] += x1 * sin
        return y

def custom_kernel(data: input_t) -> output_t:
    config, x, kv_cache = data

    f = open('mla-decode.log', 'a')

    bs = config.batch_size
    sl = config.seq_len
    pl = kv_cache.seq_len
    nh = config.n_heads
    d =  config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    wDQ  = config.Q_proj_down_weight                        # [dq, d]
    wDKV = config.KV_proj_down_weight                       # [dkv + drope, d]
    wUQ  = config.Q_proj_up_weight                          # [nh * (dnope + drope), dq]
    wUKV = config.KV_proj_up_weight                         # [nh * (dnope + dv), dkv]
    wO   = config.wo_weight                                 # [d, nh * dv]

    qkv = torch.empty((bs, sl, nh, dv), dtype=torch.bfloat16, device='cuda')
    output = torch.empty((bs, sl, d), dtype=torch.bfloat16, device='cuda')

    for i_bs in range(bs):
        x_bs = x[i_bs, :, :]                                # [sl, d]

        qD_bs = x_bs @ wDQ.T                                # [sl, dq]
        qU_bs = qD_bs @ wUQ.T                               # [sl, nh * (dnope + drope)]
        qU_bs = qU_bs.reshape(sl, nh, dnope + drope)
        qC_bs = qU_bs[:, :, :dnope]                         # [sl, nh, dnope]
        qR_bs = qU_bs[:, :, dnope:]                         # [sl, nh, drope]
        qR_bs = rope(qR_bs, pl)                             # [sl, nh, drope]

        q_bs  = torch.cat([qC_bs, qR_bs], dim=-1)           # [sl, nh, dnope + drope]

        kvD_bs = x_bs @ wDKV.T                              # [sl, dkv + drope]
        kv_cache.data[i_bs, pl:pl+sl, :] = kvD_bs
        kv_cache.seq_len += sl
        kvD_bs = kv_cache.data[i_bs, :pl+sl, :]             # [pl+sl, dkv + drope]
        kvD_bs = kvD_bs.reshape(pl+sl, dkv + drope)
        kvC_bs = kvD_bs[:, :dkv]                            # [pl+sl, dkv]
        kR_bs = kvD_bs[:, dkv:]                             # [pl+sl, drope]
        kR_bs = rope(kR_bs, 0)                              # [pl+sl, drope]
        kR_bs = kR_bs.unsqueeze(1).expand(pl+sl, nh, drope) # [pl+sl, nh, drope]
        kvU_bs = kvC_bs @ wUKV.T                            # [pl+sl, nh * (dnope + dv)]
        kvU_bs = kvU_bs.reshape(pl+sl, nh, dnope + dv)
        kC_bs = kvU_bs[:, :, :dnope]                        # [pl+sl, nh, dnope]
        k_bs = torch.cat([kC_bs, kR_bs], dim=-1)            # [pl+sl, nh, dnope + drope]
        v_bs = kvU_bs[:, :, dnope:]                         # [pl+sl, nh, dv]

        for i_nh in range(nh):
            q_h = q_bs[:, i_nh, :]                          # [sl, dnope + drope]
            k_h = k_bs[:, i_nh, :]                          # [pl+sl, dnope + drope]
            v_h = v_bs[:, i_nh, :]                          # [pl+sl, dv]
            qk_h = q_h @ k_h.T                              # [sl, pl+sl]
            qk_h = qk_h / math.sqrt(dnope + drope)
            qk_h = torch.softmax(qk_h, dim=-1)              # [sl, pl+sl]
            qkv_h = qk_h @ v_h                              # [sl, dv]
            qkv[i_bs, :, i_nh, :] = qkv_h
        output[i_bs, :, :] = qkv[i_bs, :, :, :].reshape(sl, nh * dv) @ wO.T

    return output, kv_cache.data