import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CPP_WRAPPER = """
void mla_decode1(torch::Tensor x, torch::Tensor wDQ, torch::Tensor wDKV,
                torch::Tensor wUQ, torch::Tensor wUKV, torch::Tensor wO,
                torch::Tensor kvcache, torch::Tensor output,
                int bs, int sl, int pl, int msl, int nh,
                int d, int dq, int dkv, int dnope, int drope, int dv,
                torch::Tensor qD, torch::Tensor qU, torch::Tensor kR,
                torch::Tensor qCwUK, torch::Tensor qk,
                torch::Tensor qkkvC, torch::Tensor qkv);
void mla_decode2(torch::Tensor x, torch::Tensor wDQ, torch::Tensor wDKV,
                torch::Tensor wUQ, torch::Tensor wUKV, torch::Tensor wO,
                torch::Tensor kvcache, torch::Tensor output,
                int bs, int sl, int pl, int msl, int nh,
                int d, int dq, int dkv, int dnope, int drope, int dv,
                torch::Tensor qD, torch::Tensor qU, torch::Tensor kR,
                torch::Tensor qCwUK, torch::Tensor qk,
                torch::Tensor qkkvC, torch::Tensor qkv);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

__global__ void bfgemm_mk_nk_naive_kernel(int M, int N, int K, int lda, int ldb, int ldc,
        const __hip_bfloat16 *a, const __hip_bfloat16 *b, __hip_bfloat16 *c) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += (float)a[m * lda + k] * (float)b[n * ldb + k];
    }
    c[m * ldc + n] = (__hip_bfloat16)sum;
}
void bfgemm_mk_nk_naive(int M, int N, int K, int lda, int ldb, int ldc,
        const __hip_bfloat16 *a, const __hip_bfloat16 *b, __hip_bfloat16 *c) {
    bfgemm_mk_nk_naive_kernel<<<dim3((M+15)/16, (N+15)/16), dim3(16, 16)>>>(
        M, N, K, lda, ldb, ldc, a, b, c);
}

__global__ void bfgemm_acc_mk_nk_naive_kernel(int M, int N, int K, int lda, int ldb, int ldc,
        const __hip_bfloat16 *a, const __hip_bfloat16 *b, __hip_bfloat16 *c) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float sum = c[m * ldc + n];
    for (int k = 0; k < K; ++k) {
        sum += (float)a[m * lda + k] * (float)b[n * ldb + k];
    }
    c[m * ldc + n] = (__hip_bfloat16)sum;
}
void bfgemm_acc_mk_nk_naive(int M, int N, int K, int lda, int ldb, int ldc,
        const __hip_bfloat16 *a, const __hip_bfloat16 *b, __hip_bfloat16 *c) {
    bfgemm_acc_mk_nk_naive_kernel<<<dim3((M+15)/16, (N+15)/16), dim3(16, 16)>>>(
        M, N, K, lda, ldb, ldc, a, b, c);
}

__global__ void bfgemm_mk_kn_naive_kernel(int M, int N, int K, int lda, int ldb, int ldc,
        const __hip_bfloat16 *a, const __hip_bfloat16 *b, __hip_bfloat16 *c) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += (float)a[m * lda + k] * (float)b[k * ldb + n];
    }
    c[m * ldc + n] = (__hip_bfloat16)sum;
}
void bfgemm_mk_kn_naive(int M, int N, int K, int lda, int ldb, int ldc,
        const __hip_bfloat16 *a, const __hip_bfloat16 *b, __hip_bfloat16 *c) {
    bfgemm_mk_kn_naive_kernel<<<dim3((M+15)/16, (N+15)/16), dim3(16, 16)>>>(
        M, N, K, lda, ldb, ldc, a, b, c);
}

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
void rope_3d(int d0, int d1, int d2, int xs1, int xs2, int ys1, int ys2, int pos,
        const __hip_bfloat16 *x, __hip_bfloat16 *y) {
    // x[d0, xs1, xs2], y[d0, ys1, ys2]
    // y[d0, :d1, :d2] = RoPE(x[d0, :d1, :d2], pos)
    rope_3d_kernel<<<dim3(d0, d1), dim3(64)>>>(
        d0, d1, d2, xs1, xs2, ys1, ys2, pos, x, y);
}

__global__ void sqrtdiv_softmax_kernel(int n, int d, float div, const __hip_bfloat16 *x, __hip_bfloat16 *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float sqrt_div = sqrtf(div);
    float max_val = -INFINITY;
    for (int i = 0; i < d; i++) {
        float val = (float)x[idx * d + i] / sqrt_div;
        max_val = fmaxf(max_val, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        float val = expf((float)x[idx * d + i] / sqrt_div - max_val);
        sum += val;
    }
    for (int i = 0; i < d; i++) {
        float val = expf((float)x[idx * d + i] / sqrt_div - max_val) / sum;
        y[idx * d + i] = (__hip_bfloat16)val;
    }
}
void sqrtdiv_softmax(int n, int d, float div, __hip_bfloat16 *x) {
    // x[n, d]
    // x = softmax(x / sqrt(div))
    sqrtdiv_softmax_kernel<<<(n + 63) / 64, 64>>>(n, d, div, x, x);
}

void mla_decode1(torch::Tensor x, torch::Tensor wDQ, torch::Tensor wDKV,
                torch::Tensor wUQ, torch::Tensor wUKV, torch::Tensor wO,
                torch::Tensor kvcache, torch::Tensor output,
                int bs, int sl, int pl, int msl, int nh,
                int d, int dq, int dkv, int dnope, int drope, int dv,
                torch::Tensor qD, torch::Tensor qU, torch::Tensor kR,
                torch::Tensor qCwUK, torch::Tensor qk,
                torch::Tensor qkkvC, torch::Tensor qkv) {

    __hip_bfloat16 *x_ptr = (__hip_bfloat16 *)x.data_ptr();             // [bs, sl, d]
    __hip_bfloat16 *wDQ_ptr = (__hip_bfloat16 *)wDQ.data_ptr();         // [dq, d]
    __hip_bfloat16 *wDKV_ptr = (__hip_bfloat16 *)wDKV.data_ptr();       // [dkv + drope, d]
    __hip_bfloat16 *wUQ_ptr = (__hip_bfloat16 *)wUQ.data_ptr();         // [nh * (dnope + drope), dq]
    __hip_bfloat16 *wUKV_ptr = (__hip_bfloat16 *)wUKV.data_ptr();       // [nh * (dnope + dv), dkv]
    __hip_bfloat16 *wO_ptr = (__hip_bfloat16 *)wO.data_ptr();           // [d, nh * dv]

    __hip_bfloat16 *kvcache_ptr = (__hip_bfloat16 *)kvcache.data_ptr(); // [bs, msl, dkv + drope]
    __hip_bfloat16 *output_ptr = (__hip_bfloat16 *)output.data_ptr();   // [bs, sl, d]

    __hip_bfloat16 *qD_ptr = (__hip_bfloat16 *)qD.data_ptr();           // [bs, sl, dq]
    __hip_bfloat16 *qU_ptr = (__hip_bfloat16 *)qU.data_ptr();           // [bs, sl, nh * (dnope + drope)]
    __hip_bfloat16 *kR_ptr = (__hip_bfloat16 *)kR.data_ptr();           // [bs, pl+sl, drope]
    __hip_bfloat16 *qCwUK_ptr = (__hip_bfloat16 *)qCwUK.data_ptr();     // [bs, nh, sl, dkv]
    __hip_bfloat16 *qk_ptr = (__hip_bfloat16 *)qk.data_ptr();           // [bs, nh, sl, pl+sl]
    __hip_bfloat16 *qkkvC_ptr = (__hip_bfloat16 *)qkkvC.data_ptr();     // [bs, nh, sl, dkv]
    __hip_bfloat16 *qkv_ptr = (__hip_bfloat16 *)qkv.data_ptr();         // [bs, nh, sl, dv]

    // qD = x @ wDQ
    // qD[bs, sl, dq], x[bs, sl, d], wDQ[dq, d]
    // bfgemm [bs, dq] = [bs, d] * [dq, d]
    bfgemm_mk_nk_naive(bs, dq, d, d, d, dq, x_ptr, wDQ_ptr, qD_ptr);

    // qU = qD @ wUQ
    // qU[bs, sl, nh * (dnope + drope)], qD[bs, sl, dq], wUQ[nh * (dnope + drope), dq]
    // bfgemm [bs, nh * (dnope + drope)] = [bs, dq] * [nh * (dnope + drope), dq]
    bfgemm_mk_nk_naive(bs, nh*(dnope+drope), dq, dq, dq, nh*(dnope+drope), qD_ptr, wUQ_ptr, qU_ptr);

    // qU[:, :, :, dnope:] = RoPE(qU[:, :, :, dnope:], pl)
    // qU[bs, sl, nh, dnope + drope]
    // rope [bs, sl, nh, drope]
//     rope_3d(bs*nh, sl, drope, sl*(dnope+drope), dnope+drope, sl*(dnope+drope), dnope+drope, pl, qU_ptr+dnope, qU_ptr+dnope);

    // kvcache[:, pl:pl+sl, :] = x @ wDKV
    // kvcache[bs, msl, dkv + drope], x[bs, sl, d], wDKV[dkv + drope, d]
    // bfgemm [bs, dkv + drope] = [bs, d] * [dkv + drope, d]
    bfgemm_mk_nk_naive(bs, dkv+drope, d, d, d, msl*(dkv+drope), x_ptr, wDKV_ptr, kvcache_ptr + pl*(dkv+drope));

    // kR = RoPE(kvcache[:, :pl+sl, dkv:], 0)
    // kR[bs, pl+sl, drope], kvcache[bs, msl, dkv + drope]
    // rope [bs, pl+sl, drope]
//     rope_3d(bs, pl+sl, drope, msl*(dkv+drope), dkv+drope, (pl+sl)*drope, drope, 0, kvcache_ptr + dkv, kR_ptr);
/*
    // qCwUK = qU[:, :, :dnope] @ wUKV[:, :dnope, :], where batch is nh
    // qCwUK[bs, nh, sl, dkv], qU[bs, sl, nh, dnope + drope], wUKV[nh, dnope + dv, dkv]
    // batched_bfgemm [bs, nh, dkv] = [bs, nh, dnope] * [nh, dnope, dkv]
    for (int i = 0; i < nh; i++) {
        bfgemm_mk_kn_naive(bs, dkv, dnope, nh*(dnope+drope), dkv, nh*dkv,
            qU_ptr + i*(dnope+drope), wUKV_ptr + i*(dnope+dv)*dkv, qCwUK_ptr + i*dkv);
    }

    // qk = qCwUK @ kvcache[:, :pl+sl, :dkv], where batch is bs
    // qk[bs, nh, sl, pl+sl], qCwUK[bs, nh, sl, dkv], kvcache[bs, msl, dkv + drope]
    // batched_bfgemm [bs, nh, pl+sl] = [bs, nh, dkv] * [bs, pl+sl, dkv]
    for (int i = 0; i < bs; i++) {
        bfgemm_mk_nk_naive(nh, pl+sl, dkv, dkv, dkv+drope, pl+sl,
            qCwUK_ptr + i*nh*dkv, kvcache_ptr + i*msl*(dkv+drope), qk_ptr + i*nh*(pl+sl));
    }

    // qk += qU[:, :, dnope:] @ kR, where batch is bs
    // qk[bs, nh, sl, pl+sl], qU[bs, sl, nh, dnope + drope], kR[bs, pl+sl, drope]
    // batched_bfgemm [bs, nh, pl+sl] = [bs, nh, drope] * [bs, pl+sl, drope]
    for (int i = 0; i < bs; i++) {
        bfgemm_acc_mk_nk_naive(nh, pl+sl, drope, dnope+drope, drope, pl+sl,
            qU_ptr + i*(dnope+drope)+dnope, kR_ptr + i*(pl+sl)*drope, qk_ptr + i*nh*(pl+sl));
    }

    // qk = softmax(qk / sqrt(dnope + drope))
    // qk[bs, nh, sl, pl+sl]
    // softmax [bs, nh, pl+sl]
    sqrtdiv_softmax(bs*nh, pl+sl, dnope+drope, qk_ptr);

    // qkkvC = qk @ kvcache[:, :pl+sl, :dkv], where batch is bs
    // qkkvC[bs, nh, sl, dkv], qk[bs, nh, sl, pl+sl], kvcache[bs, msl, dkv + drope]
    // batched_bfgemm [bs, nh, dkv] = [bs, nh, pl+sl] * [bs, pl+sl, dkv] (kvcache needs transpose)
    for (int i = 0; i < bs; i++) {
        bfgemm_mk_kn_naive(nh, dkv, pl+sl, pl+sl, dkv+drope, dkv,
            qk_ptr + i*nh*(pl+sl), kvcache_ptr + i*msl*(dkv+drope), qkkvC_ptr + i*nh*dkv);
    }

    // qkv = qkkvC @ wUKV[:, dnope:, :], where batch is nh
    // qkv[bs, sl, nh, dv], qkkvC[bs, sl, nh, dkv], wUKV[nh, dnope + dv, dkv]
    // batched_bfgemm [bs, nh, dv] = [bs, nh, dkv] * [nh, dv, dkv]
    for (int i = 0; i < nh; i++) {
        bfgemm_mk_kn_naive(bs, dv, dkv, nh*dkv, dkv, nh*dv,
            qkkvC_ptr + i*nh*dkv, wUKV_ptr + i*(dnope+dv)*dkv+dnope*dkv, qkv_ptr + i*nh*dv);
    }

    // output = qkv @ wO
    // output[bs, sl, d], qkv[bs, sl, nh, dv], wO[d, nh * dv]
    // bfgemm [bs, sl, d] = [bs, nh * dv] * [d, nh * dv]
    bfgemm_mk_nk_naive(bs, d, nh*dv, nh*dv, nh*dv, d, qkv_ptr, wO_ptr, output_ptr);
*/
}

void mla_decode2(torch::Tensor x, torch::Tensor wDQ, torch::Tensor wDKV,
                torch::Tensor wUQ, torch::Tensor wUKV, torch::Tensor wO,
                torch::Tensor kvcache, torch::Tensor output,
                int bs, int sl, int pl, int msl, int nh,
                int d, int dq, int dkv, int dnope, int drope, int dv,
                torch::Tensor qD, torch::Tensor qU, torch::Tensor kR,
                torch::Tensor qCwUK, torch::Tensor qk,
                torch::Tensor qkkvC, torch::Tensor qkv) {

    __hip_bfloat16 *x_ptr = (__hip_bfloat16 *)x.data_ptr();             // [bs, sl, d]
    __hip_bfloat16 *wDQ_ptr = (__hip_bfloat16 *)wDQ.data_ptr();         // [dq, d]
    __hip_bfloat16 *wDKV_ptr = (__hip_bfloat16 *)wDKV.data_ptr();       // [dkv + drope, d]
    __hip_bfloat16 *wUQ_ptr = (__hip_bfloat16 *)wUQ.data_ptr();         // [nh * (dnope + drope), dq]
    __hip_bfloat16 *wUKV_ptr = (__hip_bfloat16 *)wUKV.data_ptr();       // [nh * (dnope + dv), dkv]
    __hip_bfloat16 *wO_ptr = (__hip_bfloat16 *)wO.data_ptr();           // [d, nh * dv]

    __hip_bfloat16 *kvcache_ptr = (__hip_bfloat16 *)kvcache.data_ptr(); // [bs, msl, dkv + drope]
    __hip_bfloat16 *output_ptr = (__hip_bfloat16 *)output.data_ptr();   // [bs, sl, d]

    __hip_bfloat16 *qD_ptr = (__hip_bfloat16 *)qD.data_ptr();           // [bs, sl, dq]
    __hip_bfloat16 *qU_ptr = (__hip_bfloat16 *)qU.data_ptr();           // [bs, sl, nh * (dnope + drope)]
    __hip_bfloat16 *kR_ptr = (__hip_bfloat16 *)kR.data_ptr();           // [bs, pl+sl, drope]
    __hip_bfloat16 *qCwUK_ptr = (__hip_bfloat16 *)qCwUK.data_ptr();     // [bs, nh, sl, dkv]
    __hip_bfloat16 *qk_ptr = (__hip_bfloat16 *)qk.data_ptr();           // [bs, nh, sl, pl+sl]
    __hip_bfloat16 *qkkvC_ptr = (__hip_bfloat16 *)qkkvC.data_ptr();     // [bs, nh, sl, dkv]
    __hip_bfloat16 *qkv_ptr = (__hip_bfloat16 *)qkv.data_ptr();         // [bs, nh, sl, dv]
/*
    // qD = x @ wDQ
    // qD[bs, sl, dq], x[bs, sl, d], wDQ[dq, d]
    // bfgemm [bs, dq] = [bs, d] * [dq, d]
    bfgemm_mk_nk_naive(bs, dq, d, d, d, dq, x_ptr, wDQ_ptr, qD_ptr);

    // qU = qD @ wUQ
    // qU[bs, sl, nh * (dnope + drope)], qD[bs, sl, dq], wUQ[nh * (dnope + drope), dq]
    // bfgemm [bs, nh * (dnope + drope)] = [bs, dq] * [nh * (dnope + drope), dq]
    bfgemm_mk_nk_naive(bs, nh*(dnope+drope), dq, dq, dq, nh*(dnope+drope), qD_ptr, wUQ_ptr, qU_ptr);

    // qU[:, :, :, dnope:] = RoPE(qU[:, :, :, dnope:], pl)
    // qU[bs, sl, nh, dnope + drope]
    // rope [bs, sl, nh, drope]
    rope_3d(bs*nh, sl, drope, sl*(dnope+drope), dnope+drope, sl*(dnope+drope), dnope+drope, pl, qU_ptr+dnope, qU_ptr+dnope);

    // kvcache[:, pl:pl+sl, :] = x @ wDKV
    // kvcache[bs, msl, dkv + drope], x[bs, sl, d], wDKV[dkv + drope, d]
    // bfgemm [bs, dkv + drope] = [bs, d] * [dkv + drope, d]
    bfgemm_mk_nk_naive(bs, dkv+drope, d, d, d, msl*(dkv+drope), x_ptr, wDKV_ptr, kvcache_ptr + pl*(dkv+drope));

    // kR = RoPE(kvcache[:, :pl+sl, dkv:], 0)
    // kR[bs, pl+sl, drope], kvcache[bs, msl, dkv + drope]
    // rope [bs, pl+sl, drope]
    rope_3d(bs, pl+sl, drope, msl*(dkv+drope), dkv+drope, (pl+sl)*drope, drope, 0, kvcache_ptr + dkv, kR_ptr);
*/
    // qCwUK = qU[:, :, :dnope] @ wUKV[:, :dnope, :], where batch is nh
    // qCwUK[bs, nh, sl, dkv], qU[bs, sl, nh, dnope + drope], wUKV[nh, dnope + dv, dkv]
    // batched_bfgemm [bs, nh, dkv] = [bs, nh, dnope] * [nh, dnope, dkv]
    for (int i = 0; i < nh; i++) {
        bfgemm_mk_kn_naive(bs, dkv, dnope, nh*(dnope+drope), dkv, nh*dkv,
            qU_ptr + i*(dnope+drope), wUKV_ptr + i*(dnope+dv)*dkv, qCwUK_ptr + i*dkv);
    }

    // qk = qCwUK @ kvcache[:, :pl+sl, :dkv], where batch is bs
    // qk[bs, nh, sl, pl+sl], qCwUK[bs, nh, sl, dkv], kvcache[bs, msl, dkv + drope]
    // batched_bfgemm [bs, nh, pl+sl] = [bs, nh, dkv] * [bs, pl+sl, dkv]
    for (int i = 0; i < bs; i++) {
        bfgemm_mk_nk_naive(nh, pl+sl, dkv, dkv, dkv+drope, pl+sl,
            qCwUK_ptr + i*nh*dkv, kvcache_ptr + i*msl*(dkv+drope), qk_ptr + i*nh*(pl+sl));
    }

    // qk += qU[:, :, dnope:] @ kR, where batch is bs
    // qk[bs, nh, sl, pl+sl], qU[bs, sl, nh, dnope + drope], kR[bs, pl+sl, drope]
    // batched_bfgemm [bs, nh, pl+sl] = [bs, nh, drope] * [bs, pl+sl, drope]
    for (int i = 0; i < bs; i++) {
        bfgemm_acc_mk_nk_naive(nh, pl+sl, drope, dnope+drope, drope, pl+sl,
            qU_ptr + i*(dnope+drope)+dnope, kR_ptr + i*(pl+sl)*drope, qk_ptr + i*nh*(pl+sl));
    }

    // qk = softmax(qk / sqrt(dnope + drope))
    // qk[bs, nh, sl, pl+sl]
    // softmax [bs, nh, pl+sl]
    sqrtdiv_softmax(bs*nh, pl+sl, dnope+drope, qk_ptr);

    // qkkvC = qk @ kvcache[:, :pl+sl, :dkv], where batch is bs
    // qkkvC[bs, nh, sl, dkv], qk[bs, nh, sl, pl+sl], kvcache[bs, msl, dkv + drope]
    // batched_bfgemm [bs, nh, dkv] = [bs, nh, pl+sl] * [bs, pl+sl, dkv] (kvcache needs transpose)
    for (int i = 0; i < bs; i++) {
        bfgemm_mk_kn_naive(nh, dkv, pl+sl, pl+sl, dkv+drope, dkv,
            qk_ptr + i*nh*(pl+sl), kvcache_ptr + i*msl*(dkv+drope), qkkvC_ptr + i*nh*dkv);
    }

    // qkv = qkkvC @ wUKV[:, dnope:, :], where batch is nh
    // qkv[bs, sl, nh, dv], qkkvC[bs, sl, nh, dkv], wUKV[nh, dnope + dv, dkv]
    // batched_bfgemm [bs, nh, dv] = [bs, nh, dkv] * [nh, dv, dkv]
    for (int i = 0; i < nh; i++) {
        bfgemm_mk_kn_naive(bs, dv, dkv, nh*dkv, dkv, nh*dv,
            qkkvC_ptr + i*nh*dkv, wUKV_ptr + i*(dnope+dv)*dkv+dnope*dkv, qkv_ptr + i*nh*dv);
    }

    // output = qkv @ wO
    // output[bs, sl, d], qkv[bs, sl, nh, dv], wO[d, nh * dv]
    // bfgemm [bs, sl, d] = [bs, nh * dv] * [d, nh * dv]
    bfgemm_mk_nk_naive(bs, d, nh*dv, nh*dv, nh*dv, d, qkv_ptr, wO_ptr, output_ptr);
}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='mla_decode',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['mla_decode1', 'mla_decode2'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O3", "-w"],
)

def rope(x: torch.Tensor, pos: int) -> torch.Tensor:

    n, sl, d = x.shape

    theta = torch.arange(d // 2, dtype=torch.bfloat16, device='cuda')
    theta = 10000 ** (-2 * theta / d)
    idx = torch.arange(pos, pos + sl, dtype=torch.bfloat16, device='cuda')
    idx_theta = torch.einsum('s,d->sd', idx, theta)
    cos = torch.cos(idx_theta).to(torch.bfloat16)
    sin = torch.sin(idx_theta).to(torch.bfloat16)

    y = torch.empty_like(x)
    x1 = x[:, :, :d//2]
    x2 = x[:, :, d//2:]
    y[:, :, :d//2] = x1 * cos - x2 * sin
    y[:, :, d//2:] = x2 * cos + x1 * sin
    return y

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

    if (sl != 1):
        return

    # x                               # [bs, sl, d]
    wDQ  = config.Q_proj_down_weight  # [dq, d]
    wDKV = config.KV_proj_down_weight # [dkv + drope, d]
    wUQ  = config.Q_proj_up_weight    # [nh * (dnope + drope), dq]
    wUKV = config.KV_proj_up_weight   # [nh * (dnope + dv), dkv]
    wO   = config.wo_weight           # [d, nh * dv]
    kvcache = kv_cache.data           # [bs, msl, dkv + drope]
    output = torch.empty((bs, sl, d), dtype=torch.bfloat16, device='cuda') # [bs, sl, d]

    qD = torch.empty((bs, sl, dq), dtype=torch.bfloat16, device='cuda') # [bs, sl, dq]
    qU = torch.empty((bs, sl, nh * (dnope + drope)), dtype=torch.bfloat16, device='cuda') # [bs, sl, nh * (dnope + drope)]
    kR = torch.empty((bs, pl+sl, drope), dtype=torch.bfloat16, device='cuda') # [bs, pl+sl, drope]
    qCwUK = torch.empty((bs, nh, sl, dkv), dtype=torch.bfloat16, device='cuda') # [bs, nh, sl, dkv]
    qk = torch.empty((bs, nh, sl, pl+sl), dtype=torch.bfloat16, device='cuda') # [bs, nh, sl, pl+sl]
    qkkvC = torch.empty((bs, nh, sl, dkv), dtype=torch.bfloat16, device='cuda') # [bs, nh, sl, dkv]
    qkv = torch.empty((bs, sl, nh, dv), dtype=torch.bfloat16, device='cuda') # [bs, nh, sl, dv]

    module.mla_decode1(x, wDQ, wDKV, wUQ, wUKV, wO, kvcache, output,
        bs, sl, pl, msl, nh, d, dq, dkv, dnope, drope, dv,
        qD, qU, kR, qCwUK, qk, qkkvC, qkv)

    # rope_3d(bs*nh, sl, drope, sl*(dnope+drope), dnope+drope, sl*(dnope+drope), dnope+drope, pl, qU_ptr+dnope, qU_ptr+dnope);
    # rope_3d(bs, pl+sl, drope, msl*(dkv+drope), dkv+drope, (pl+sl)*drope, drope, 0, kvcache_ptr + dkv, kR_ptr);
    qU = qU.reshape(bs*nh, sl, dnope + drope)
    qU[:, :, dnope:] = rope(qU[:, :, dnope:], pl)
    qU = qU.reshape(bs, sl, nh * (dnope + drope))
    kR = rope(kvcache[:, :pl+sl, dkv:], 0)

    module.mla_decode2(x, wDQ, wDKV, wUQ, wUKV, wO, kvcache, output,
        bs, sl, pl, msl, nh, d, dq, dkv, dnope, drope, dv,
        qD, qU, kR, qCwUK, qk, qkkvC, qkv)

    kv_cache.seq_len += sl
    return output, kv_cache.data
