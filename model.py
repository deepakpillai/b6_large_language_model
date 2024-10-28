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
import torch.utils.checkpoint as checkpoint

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.NUM_HEADS
        self.attention_head_size = int(config.EMBED_SIZE / config.NUM_HEADS)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.layer_norm = LayerNorm(config.EMBED_SIZE)
        self.qkv = nn.Linear(config.EMBED_SIZE, 3 * self.all_head_size, bias=False)
        self.dense = nn.Linear(config.EMBED_SIZE, config.EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        self.scale = math.sqrt(self.attention_head_size)

    def forward(self, hidden_states, attention_mask=None):
        normalized_hidden_states = self.layer_norm(hidden_states)
        batch_size, seq_length, _ = normalized_hidden_states.size()
        
        qkv = self.qkv(normalized_hidden_states)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, 
                       h=self.num_attention_heads)
        
        def attention_forward(qkv, attention_mask):
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                qkv, indices, cu_seqlens, max_seqlen = unpad_input(qkv, attention_mask)
                context_layer = flash_attn_qkvpacked_func(
                    qkv, 
                    cu_seqlens=cu_seqlens, 
                    max_seqlen=max_seqlen, 
                    dropout_p=self.dropout.p if self.training else 0.0,
                    softmax_scale=self.scale
                )
                context_layer = pad_input(context_layer, indices, batch_size, seq_length)
            else:
                context_layer = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    softmax_scale=self.scale
                )
            return context_layer
        
        if self.training:
            context_layer = checkpoint.checkpoint(attention_forward, qkv, attention_mask)
        else:
            context_layer = attention_forward(qkv, attention_mask)
            
        context_layer = rearrange(context_layer, 'b s h d -> b s (h d)')
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        
        return attention_output + hidden_states


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = LayerNorm(config.EMBED_SIZE)
        self.dense1 = nn.Linear(config.EMBED_SIZE, config.HIDDEN_DIM)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(config.HIDDEN_DIM, config.EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, hidden_states):
        def ffn_forward(hidden_states):
            normalized_hidden_states = self.layer_norm(hidden_states)
            hidden_states_inner = self.dense1(normalized_hidden_states)
            hidden_states_inner = self.intermediate_act_fn(hidden_states_inner)
            hidden_states_inner = self.dense2(hidden_states_inner)
            hidden_states_inner = self.dropout(hidden_states_inner)
            return hidden_states_inner
        
        if self.training:
            hidden_states_inner = checkpoint.checkpoint(ffn_forward, hidden_states)
        else:
            hidden_states_inner = ffn_forward(hidden_states)
            
        return hidden_states_inner + hidden_states


class FlashTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = FlashMultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.use_checkpoint = True

    def forward(self, x, attention_mask=None):
        def layer_forward(x, attention_mask):
            x = self.attention(x, attention_mask)
            x = self.ffn(x)
            return x
            
        if self.training and self.use_checkpoint:
            return checkpoint.checkpoint(layer_forward, x, attention_mask)
        return layer_forward(x, attention_mask)

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