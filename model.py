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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.NUM_HEADS
        self.attention_head_size = int(config.EMBED_SIZE / config.NUM_HEADS)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.EMBED_SIZE, self.all_head_size)
        self.key = nn.Linear(config.EMBED_SIZE, self.all_head_size)
        self.value = nn.Linear(config.EMBED_SIZE, self.all_head_size)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.dense = nn.Linear(config.EMBED_SIZE, config.EMBED_SIZE)
        self.layer_norm = LayerNorm(config.EMBED_SIZE)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(attention_output + hidden_states)
        
        return attention_output

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.EMBED_SIZE, config.HIDDEN_DIM)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(config.HIDDEN_DIM, config.EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.layer_norm = LayerNorm(config.EMBED_SIZE)

    def forward(self, hidden_states):
        hidden_states_inner = self.dense1(hidden_states)
        hidden_states_inner = self.intermediate_act_fn(hidden_states_inner)
        hidden_states_inner = self.dense2(hidden_states_inner)
        hidden_states_inner = self.dropout(hidden_states_inner)
        hidden_states = self.layer_norm(hidden_states + hidden_states_inner)
        return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
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
        
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.NUM_LAYERS)
        ])
        
        self.layer_norm = LayerNorm(config.EMBED_SIZE)
        self.fc_out = nn.Linear(config.EMBED_SIZE, config.VOCAB_SIZE)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
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
            
        hidden_states = self.layer_norm(hidden_states)
        logits = self.fc_out(hidden_states)
        
        return logits

def create_optimizer(model, config):
    # Implementing weight decay fix
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)
    return optimizer