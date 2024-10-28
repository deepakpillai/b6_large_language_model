import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import math
import torch.nn.functional as F
import wandb  # For experiment tracking
from torch.nn import LayerNorm
import hyperparameters
import math
from optimizer import Lion
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from flash_attn.bert_padding import unpad_input, pad_input
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
from typing import Optional, Tuple

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlashMultiHeadAttention(nn.Module):
    """
    Implements Flash Attention v2 with better memory efficiency and speed.
    """
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.NUM_HEADS
        self.attention_head_size = int(config.EMBED_SIZE / config.NUM_HEADS)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(config.EMBED_SIZE, self.all_head_size, bias=False)
        self.k_proj = nn.Linear(config.EMBED_SIZE, self.all_head_size, bias=False)
        self.v_proj = nn.Linear(config.EMBED_SIZE, self.all_head_size, bias=False)
        
        self.dropout = nn.Dropout(config.DROPOUT)
        self.layer_norm = LayerNorm(config.EMBED_SIZE)
        self.out_proj = nn.Linear(config.EMBED_SIZE, config.EMBED_SIZE)
        
        # Scaling factor for better numerical stability
        self.scale = math.sqrt(self.attention_head_size)
        
        # Memory efficient attention settings
        self.head_dim = config.EMBED_SIZE // config.NUM_HEADS
        assert self.head_dim * config.NUM_HEADS == config.EMBED_SIZE, "embed_dim must be divisible by num_heads"

    def _reshape_for_flash_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input tensor for flash attention."""
        batch_size, seq_length, _ = x.size()
        x = rearrange(x, 'b s (h d) -> b s h d', h=self.num_attention_heads)
        return x

    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                causal: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Flash Attention v2 optimizations.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, embed_dim)
            attention_mask: Optional mask tensor of shape (batch_size, seq_length)
            causal: Whether to apply causal masking
            
        Returns:
            output: Transformed tensor of shape (batch_size, seq_length, embed_dim)
            attention_weights: Optional attention weights for visualization
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Apply layer normalization first (pre-norm)
        normed_hidden_states = self.layer_norm(hidden_states)
        
        # Generate Q, K, V projections
        def qkv_forward(hidden_states):
            q = self._reshape_for_flash_attention(self.q_proj(hidden_states))
            k = self._reshape_for_flash_attention(self.k_proj(hidden_states))
            v = self._reshape_for_flash_attention(self.v_proj(hidden_states))
            return q, k, v
        
        # Apply checkpointing for memory efficiency during training
        if self.training:
            q, k, v = checkpoint.checkpoint(qkv_forward, normed_hidden_states)
        else:
            q, k, v = qkv_forward(normed_hidden_states)
        
        # Pack QKV for flash attention
        qkv = torch.stack([q, k, v], dim=2)  # [batch_size, seq_length, 3, num_heads, head_dim]
        
        # Handle attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            
            # Unpad input if using attention mask
            qkv_unpad, indices, cu_seqlens, max_seqlen = unpad_input(qkv, attention_mask)
            
            # Apply flash attention
            context_layer = flash_attn_qkvpacked_func(
                qkv_unpad,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=causal
            )
            
            # Pad output back
            context_layer = pad_input(context_layer, indices, batch_size, seq_length)
        else:
            # Direct flash attention if no mask
            context_layer = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=causal
            )
        
        # Reshape and apply output projection
        context_layer = rearrange(context_layer, 'b s h d -> b s (h d)')
        output = self.out_proj(context_layer)
        output = self.dropout(output)
        
        return output


class FeedForward(nn.Module):
    """
    Enhanced feed-forward network with better memory efficiency.
    """
    def __init__(self, config):
        super().__init__()
        self.layer_norm = LayerNorm(config.EMBED_SIZE)
        
        # Use intermediate size multiplier for better capacity
        self.intermediate_size = config.HIDDEN_DIM
        
        self.fc1 = nn.Linear(config.EMBED_SIZE, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, config.EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        def ff_forward(x):
            x = self.layer_norm(x)
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
            return self.dropout(x)
        
        if self.training:
            return checkpoint.checkpoint(ff_forward, hidden_states)
        return ff_forward(hidden_states)


class FlashTransformerLayer(nn.Module):
    """
    Enhanced transformer layer with Flash Attention v2.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = FlashMultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.use_checkpoint = True

    def forward(self, x, attention_mask=None):
        # Residual connection for attention
        attn_output = self.attention(x, attention_mask)
        x = x + attn_output
        
        # Residual connection for feed-forward
        ff_output = self.ffn(x)
        x = x + ff_output

        return x

class ImprovedTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBED_SIZE)
        self.position_embedding = nn.Embedding(config.SEQ_LENGTH, config.EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # Use gradient checkpointing for embeddings
        self.gradient_checkpointing = True
        
        self.layers = nn.ModuleList([
            FlashTransformerLayer(config) for _ in range(config.NUM_LAYERS)
        ])
        
        self.final_layer_norm = LayerNorm(config.EMBED_SIZE)
        self.fc_out = nn.Linear(config.EMBED_SIZE, config.VOCAB_SIZE)
        
        self.apply(self._init_weights)
        self.set_param_groups()
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            else:
                module.weight.data.normal_(mean=0.0, std=0.02)
            
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def set_param_groups(self):
        self.param_groups = {
            'embeddings': list(self.embedding.parameters()) + list(self.position_embedding.parameters()),
            'attention': [],
            'ffn': [],
            'layer_norm': [],
            'output': list(self.fc_out.parameters())
        }
        
        for layer in self.layers:
            self.param_groups['attention'].extend(list(layer.attention.parameters()))
            self.param_groups['ffn'].extend(list(layer.ffn.parameters()))
            self.param_groups['layer_norm'].extend([
                p for name, p in layer.named_parameters() if 'layer_norm' in name
            ])

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
        for layer in self.layers:
            layer.use_checkpoint = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
        for layer in self.layers:
            layer.use_checkpoint = False

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        def create_embeddings(input_ids, position_ids):
            inputs_embeds = self.embedding(input_ids)
            position_embeddings = self.position_embedding(position_ids)
            return inputs_embeds + position_embeddings
        
        # Use checkpointing for embeddings during training
        if self.training and self.gradient_checkpointing:
            hidden_states = checkpoint.checkpoint(create_embeddings, input_ids, position_ids)
        else:
            hidden_states = create_embeddings(input_ids, position_ids)
            
        hidden_states = self.dropout(hidden_states)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.fc_out(hidden_states)
        
        return logits

def create_optimizer(model, config):
    """
    Creates and returns a Lion optimizer with weight decay fix.
    """
    # Implementing weight decay fix (similar to AdamW)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.1
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = Lion(
        optimizer_grouped_parameters,
        lr=config.LEARNING_RATE,
        beta1=0.95,
        beta2=0.98,
        weight_decay=0.1
    )
    return optimizer