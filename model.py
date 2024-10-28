import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing import Optional, Tuple, Union
from flash_attn import flash_attn_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from rotaryembedding import RotaryEmbedding, apply_rotary_pos_emb

class ModelConfig:
    """Configuration class for validation."""
    @staticmethod
    def validate_config(config):
        assert config.EMBED_SIZE % config.NUM_HEADS == 0, \
            f"Embedding size {config.EMBED_SIZE} not divisible by number of heads {config.NUM_HEADS}"
        assert hasattr(config, 'MAX_POSITION_EMBEDDINGS'), \
            "Config must define MAX_POSITION_EMBEDDINGS"
        assert config.SEQ_LENGTH <= config.MAX_POSITION_EMBEDDINGS, \
            f"Sequence length {config.SEQ_LENGTH} exceeds maximum position embeddings {config.MAX_POSITION_EMBEDDINGS}"
        return True

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.num_attention_heads = config.NUM_HEADS
        self.attention_head_size = int(config.EMBED_SIZE / config.NUM_HEADS)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Initialize projections with configurable bias
        self.q_proj = nn.Linear(config.EMBED_SIZE, self.all_head_size, 
                               bias=config.USE_BIAS_IN_ATTN)
        self.k_proj = nn.Linear(config.EMBED_SIZE, self.all_head_size, 
                               bias=config.USE_BIAS_IN_ATTN)
        self.v_proj = nn.Linear(config.EMBED_SIZE, self.all_head_size, 
                               bias=config.USE_BIAS_IN_ATTN)
        
        # Separate dropouts for attention and output
        self.attention_dropout = nn.Dropout(config.ATTENTION_DROPOUT)
        self.output_dropout = nn.Dropout(config.DROPOUT)
        
        # Layer norm for pre-norm architecture
        if config.PRE_NORM:
            self.layer_norm = LayerNorm(config.EMBED_SIZE)
            
        self.out_proj = nn.Linear(config.EMBED_SIZE, config.EMBED_SIZE,
                                 bias=config.USE_BIAS_IN_ATTN)
        
        # Register rotary embeddings
        self.register_buffer(
            "rotary_emb",
            RotaryEmbedding(
                self.attention_head_size,
                max_position_embeddings=config.MAX_POSITION_EMBEDDINGS,
            ).compute_freqs_cis(config.MAX_POSITION_EMBEDDINGS),
            persistent=False
        )
        
        self.scale = self.attention_head_size ** -0.5
        self.config = config
    
    def _reshape_for_flash_attention(self, x):
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.transpose(1, 2)  # (batch, heads, seq_length, head_dim)
    
    def _maybe_pad_sequence(self, seq_length: int) -> int:
        """Ensure sequence length is a multiple of config's PAD_TO_MULTIPLE_OF."""
        if self.config.PAD_TO_MULTIPLE_OF > 0:
            pad_len = (self.config.PAD_TO_MULTIPLE_OF - 
                      seq_length % self.config.PAD_TO_MULTIPLE_OF) % self.config.PAD_TO_MULTIPLE_OF
            return pad_len
        return 0

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        
        # Apply pre-norm if configured
        if self.config.PRE_NORM:
            hidden_states = self.layer_norm(hidden_states)
            
        # Handle padding for Flash Attention
        pad_len = self._maybe_pad_sequence(seq_length)
        
        def qkv_forward(hidden_states):
            q = self._reshape_for_flash_attention(self.q_proj(hidden_states))
            k = self._reshape_for_flash_attention(self.k_proj(hidden_states))
            v = self._reshape_for_flash_attention(self.v_proj(hidden_states))
            
            # Apply rotary embeddings
            cos, sin = self.rotary_emb[:seq_length], self.rotary_emb[:seq_length]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            return q.contiguous(), k.contiguous(), v.contiguous()
        
        # Apply gradient checkpointing during training if configured
        if self.training and self.config.GRADIENT_CHECKPOINTING:
            q, k, v = checkpoint.checkpoint(qkv_forward, hidden_states)
        else:
            q, k, v = qkv_forward(hidden_states)
        
        # Pack QKV for flash attention
        qkv = torch.stack([q, k, v], dim=2)
        
        # Handle attention mask and padding
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            qkv_unpad, indices, cu_seqlens, max_seqlen = unpad_input(qkv, attention_mask)
            
            # Apply Flash Attention with version check
            flash_fn = flash_attn_qkvpacked_func
            if self.config.FLASH_ATTENTION_VERSION == "2":
                # Add any version 2 specific arguments here
                pass
                
            context_layer = flash_fn(
                qkv_unpad,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=causal
            )
            
            context_layer = pad_input(context_layer, indices, batch_size, seq_length)
        else:
            context_layer = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=causal
            )
        
        # Project output and apply dropout
        context_layer = context_layer.view(batch_size, seq_length, -1)
        output = self.out_proj(context_layer)
        output = self.output_dropout(output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Layer norm for pre-norm architecture
        if config.PRE_NORM:
            self.layer_norm = LayerNorm(config.EMBED_SIZE)
            
        self.intermediate_size = config.HIDDEN_DIM
        
        # Initialize with configurable bias
        self.fc1 = nn.Linear(config.EMBED_SIZE, self.intermediate_size,
                            bias=config.USE_BIAS_IN_FFN)
        self.fc2 = nn.Linear(self.intermediate_size, config.EMBED_SIZE,
                            bias=config.USE_BIAS_IN_FFN)
        
        self.dropout = nn.Dropout(config.DROPOUT)
        self.activation = nn.GELU()
        self.config = config
        
        # Initialize weights
        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, std=0.02)
            nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, hidden_states):
        def ff_forward(x):
            # Apply pre-norm if configured
            if self.config.PRE_NORM:
                x = self.layer_norm(x)
                
            x = self.fc1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x
        
        if self.training and self.config.GRADIENT_CHECKPOINTING:
            return checkpoint.checkpoint(ff_forward, hidden_states)
        return ff_forward(hidden_states)

class FlashTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = FlashMultiHeadAttention(config)
        self.ffn = FeedForward(config)
        
        # Initialize layer norms based on architecture choice
        if not config.PRE_NORM:
            self.attention_layer_norm = LayerNorm(config.EMBED_SIZE)
            self.ffn_layer_norm = LayerNorm(config.EMBED_SIZE)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Post-norm architecture
        if not self.config.PRE_NORM:
            # Attention block
            attn_output = self.attention(x, attention_mask)
            x = x + attn_output
            x = self.attention_layer_norm(x)
            
            # FFN block
            ff_output = self.ffn(x)
            x = x + ff_output
            x = self.ffn_layer_norm(x)
        # Pre-norm architecture (default)
        else:
            # Attention block
            attn_output = self.attention(x, attention_mask)
            x = x + attn_output
            
            # FFN block
            ff_output = self.ffn(x)
            x = x + ff_output
            
        return x


class ImprovedTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        ModelConfig.validate_config(config)
        self.config = config
        
        # Token embedding with proper initialization
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBED_SIZE)
        
        # Separate dropout rates for attention and general dropout
        self.embedding_dropout = nn.Dropout(config.DROPOUT)
        self.attention_dropout = nn.Dropout(config.ATTENTION_DROPOUT)
        
        # Initialize transformer layers
        self.layers = nn.ModuleList([
            FlashTransformerLayer(config) for _ in range(config.NUM_LAYERS)
        ])
        
        # Output head
        if config.PRE_NORM:
            self.final_layer_norm = LayerNorm(config.EMBED_SIZE)
        self.fc_out = nn.Linear(config.EMBED_SIZE, config.VOCAB_SIZE, 
                               bias=config.USE_BIAS_IN_FFN)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing if configured
        if config.GRADIENT_CHECKPOINTING:
            self.gradient_checkpointing = True
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get embeddings
        hidden_states = self.embedding(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Process through transformer layers
        for layer in self.layers:
            if self.training and self.config.GRADIENT_CHECKPOINTING:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states = checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)
        
        # Apply final layer norm if using pre-norm architecture
        if self.config.PRE_NORM:
            hidden_states = self.final_layer_norm(hidden_states)
        
        # Output projection
        logits = self.fc_out(hidden_states)
        
        return logits

def create_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Creates an optimizer with proper weight decay handling."""
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config.WEIGHT_DECAY
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    return Lion(
        optimizer_grouped_parameters,
        lr=config.LEARNING_RATE,
        beta1=0.95,
        beta2=0.98,
    )