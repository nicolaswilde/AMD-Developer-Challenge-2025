import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Submission template for DeepSeek-style Mixture of Experts using PyTorch.

    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_size]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters

    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
            - aux_data: Dictionary with auxiliary data
    """
    input_tensor, weights, config = data

    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    n_routed_experts = config["n_routed_experts"]
    n_shared_experts = config["n_shared_experts"]
    n_experts_per_token = config["n_experts_per_token"]
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]

    router = torch.empty(d_hidden, n_routed_experts, dtype=torch.float16, device='cuda')
    gate = torch.empty(d_hidden, d_expert * (n_routed_experts + n_shared_experts), dtype=torch.float16, device='cuda')
    up = torch.empty(d_hidden, d_expert * (n_routed_experts + n_shared_experts), dtype=torch.float16, device='cuda')
    down = torch.empty(d_expert * (n_routed_experts + n_shared_experts), d_hidden, dtype=torch.float16, device='cuda')
    dense = torch.empty(batch_size, seq_len, d_hidden * (n_routed_experts + n_shared_experts), dtype=torch.float16, device='cuda')
    output = torch.zeros(batch_size, seq_len, d_hidden, dtype=torch.float16, device='cuda')

    router = weights['router.weight'].t()
    for i in range(n_routed_experts):
        gate[:, i * d_expert:(i + 1) * d_expert] = weights[f'experts.{i}.0.weight']
        up[:, i * d_expert:(i + 1) * d_expert] = weights[f'experts.{i}.1.weight']
        down[i * d_expert:(i + 1) * d_expert, :] = weights[f'experts.{i}.2.weight']
    gate[:, n_routed_experts * d_expert:] = weights['shared_experts.0.weight']
    up[:, n_routed_experts * d_expert:] = weights['shared_experts.1.weight']
    down[n_routed_experts * d_expert:, :] = weights['shared_experts.2.weight']

    # score = torch.nn.functional.softmax(input_tensor @ router, dim=2)
    # top_values, top_indices = torch.topk(score, n_experts_per_token, dim=2)

    for b in range(batch_size):
        input = input_tensor[b]
        score = torch.nn.functional.softmax(input @ router, dim=1)
        hidden = (torch.nn.functional.silu(input @ gate)) * (input @ up)
        for i in range(n_routed_experts + n_shared_experts):
            dense[b, :, i * d_hidden:(i + 1) * d_hidden] = hidden[:, i * d_expert:(i + 1) * d_expert] @ down[i * d_expert:(i + 1) * d_expert, :]
        for t in range(seq_len):
            top_values, top_indices = torch.topk(score[t], n_experts_per_token)
            for i in range(n_experts_per_token):
                output[b, t] += top_values[i] * dense[b, t, top_indices[i].item() * d_hidden:(top_indices[i].item() + 1) * d_hidden]
            for i in range(n_shared_experts):
                output[b, t] += dense[b, t, (n_routed_experts + i) * d_hidden:(n_routed_experts + i + 1) * d_hidden]

    return output