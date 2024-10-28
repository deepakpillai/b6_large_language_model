import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    #     """
    #     Implements Rotary Position Embeddings (RoPE)
    #     Paper: https://arxiv.org/abs/2104.09864
    #     """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        #         """
        #         Args:
        #             x: Input tensor of shape [batch_size, seq_len, n_heads, head_dim]
        #             seq_len: Sequence length (optional)
        #         """
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )

def rotate_half(x):
    # """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    #     """
    #     Apply rotary position embeddings to q and k tensors.
        
    #     Args:
    #         q: Query tensor of shape [batch_size, seq_len, n_heads, head_dim]
    #         k: Key tensor of shape [batch_size, seq_len, n_heads, head_dim]
    #         cos: Cosine part of rotary embeddings
    #         sin: Sine part of rotary embeddings
    #     """
    #     # Ensure compatibility of shapes
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed