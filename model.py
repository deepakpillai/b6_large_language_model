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
from einops import rearrange

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.NUM_HEADS
        self.attention_head_size = int(config.EMBED_SIZE / config.NUM_HEADS)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Pre-LayerNorm. More stable training dynamics due to normalized inputs, 
        # Allows for higher learning rates,
        # Better gradient flow in deep networks
        # Reduces the risk of training instability, especially in deeper models 
        self.layer_norm = LayerNorm(config.EMBED_SIZE)
        
        # Single projection matrix for Q, K, V
        self.qkv = nn.Linear(config.EMBED_SIZE, 3 * self.all_head_size, bias=False)
        self.dense = nn.Linear(config.EMBED_SIZE, config.EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # Scaling factor for attention
        self.scale = math.sqrt(self.attention_head_size)

    def forward(self, hidden_states, attention_mask=None):
        # Apply LayerNorm first (Pre-LayerNorm)
        normalized_hidden_states = self.layer_norm(hidden_states)
        
        batch_size, seq_length, _ = normalized_hidden_states.size()
        
        # Project to Q, K, V in one go
        qkv = self.qkv(normalized_hidden_states)
        
        # Reshape qkv for Flash Attention
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, 
                       h=self.num_attention_heads)
        
        # Handle attention mask for Flash Attention
        if attention_mask is not None:
            # Convert mask to boolean
            attention_mask = attention_mask.bool()
            # Unpad input and mask for Flash Attention
            qkv, indices, cu_seqlens, max_seqlen = unpad_input(qkv, attention_mask)
            # Apply Flash Attention
            context_layer = flash_attn_qkvpacked_func(
                qkv, 
                cu_seqlens=cu_seqlens, 
                max_seqlen=max_seqlen, 
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale
            )
            # Pad output back
            context_layer = pad_input(context_layer, indices, batch_size, seq_length)
        else:
            # If no mask, reshape and apply Flash Attention directly
            context_layer = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale
            )
        # Reshape output
        context_layer = rearrange(context_layer, 'b s h d -> b s (h d)')
        # Project back to embedding dimension
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        
        # Residual connection
        return attention_output + hidden_states

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Pre-LayerNorm
        self.layer_norm = LayerNorm(config.EMBED_SIZE)
        
        self.dense1 = nn.Linear(config.EMBED_SIZE, config.HIDDEN_DIM)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(config.HIDDEN_DIM, config.EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, hidden_states):
        # Apply LayerNorm first (Pre-LayerNorm)
        normalized_hidden_states = self.layer_norm(hidden_states)
        
        hidden_states_inner = self.dense1(normalized_hidden_states)
        hidden_states_inner = self.intermediate_act_fn(hidden_states_inner)
        hidden_states_inner = self.dense2(hidden_states_inner)
        hidden_states_inner = self.dropout(hidden_states_inner)
        
        # Residual connection
        return hidden_states_inner + hidden_states

class FlashTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = FlashMultiHeadAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x, attention_mask=None):
        x = self.attention(x, attention_mask)
        x = self.ffn(x)
        return x


class ImprovedTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBED_SIZE)
        self.position_embedding = nn.Embedding(config.SEQ_LENGTH, config.EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # Use Flash Attention Transformer layers
        self.layers = nn.ModuleList([
            FlashTransformerLayer(config) for _ in range(config.NUM_LAYERS)
        ])
        
        # Final layer norm - still useful for output stability
        self.final_layer_norm = LayerNorm(config.EMBED_SIZE)
        self.fc_out = nn.Linear(config.EMBED_SIZE, config.VOCAB_SIZE)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Add param groups for different learning rates
        self.set_param_groups()
    
    def set_param_groups(self):
        """Group parameters for different learning rates"""
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
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Use Kaiming initialization for linear layers
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            else:
                module.weight.data.normal_(mean=0.0, std=0.02)
            
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        inputs_embeds = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        hidden_states = inputs_embeds + position_embeddings
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