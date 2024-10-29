# B6 Large Language Model (LLM) 🚀

B6 is a high-performance language model implementation that pushes the boundaries of efficient transformer architecture. Built with PyTorch and optimized for modern GPUs, it combines cutting-edge techniques like Flash Attention, Rotary Position Embeddings (RoPE), and advanced memory management to deliver a powerful, scalable training framework for language models.

## ⚡ Core Features

### Advanced Architecture
- **Flash Attention Integration** with version-specific optimizations
- **Rotary Position Embeddings (RoPE)** for enhanced positional understanding
- **Flexible Normalization** with support for both Pre-LN and Post-LN architectures
- **Optimized Multi-Head Attention** with configurable bias and scaling
- **Adaptive Architecture** that scales from RTX 3060 to A100 configurations

### Memory Optimization Suite
- **Dynamic Memory Management** with automatic cleanup and monitoring
- **Smart Gradient Checkpointing** for optimal memory-performance trade-off
- **Gradient Accumulation** with configurable steps
- **Mixed Precision Training** using PyTorch AMP
- **Adaptive Batch Sizing** with automatic OOM recovery

### Advanced Training Components
- **Lion Optimizer** - Evolutionary optimization algorithm with proper weight decay handling
- **Optimized LLM Scheduler** featuring:
  - Linear warmup with cosine decay
  - Configurable minimum LR ratio
  - Dynamic warmup scheduling
- **Intelligent Checkpointing** with validation-based model selection

### Robust Data Pipeline
- **Streaming Dataset Support** for efficient memory usage
- **Multiple Dataset Compatibility**:
  - C4
  - OpenWebText2
  - RedPajama
  - OSCAR
  - The Stack
  - Books3
- **Efficient Data Loading** with prefetching and persistent workers

### Realistic Expectations
- With the current setup, you might achieve: 
  - Performance similar to GPT-2 Small/Medium
  - Good text generation capabilities
  - Basic language understanding
  - Limited reasoning abilities

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch ninja packaging transformers datasets wandb tqdm numpy zstandard flash-attn --no-build-isolation
```

### Training the Model

1. Configure your hyperparameters in `hyperparameters.py`:
```python
class Config:
    VOCAB_SIZE = 50257  # GPT-2 vocabulary size
    EMBED_SIZE = 1024
    NUM_HEADS = 16
    NUM_LAYERS = 24
    # ... (other parameters)
```

2. Start training:
```python
python app.py
```

## 💡 Model Architecture

The B6 model implements a state-of-the-art transformer architecture with several modern optimizations:

### Core Components

- **Flash Attention Mechanism**: Implements an optimized attention mechanism using the Flash Attention algorithm, providing significant memory and computational efficiency improvements over traditional attention mechanisms. The implementation supports both Flash Attention v1 and v2.

```python
# Memory-efficient attention computation
with torch.amp.autocast(device_type='cuda'):
    flash_fn = flash_attn_qkvpacked_func
    context_layer = flash_fn(
        qkv_unpad,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        dropout_p=self.attention_dropout.p if self.training else 0.0,
        softmax_scale=self.scale,
        causal=causal
    )
```

- **Rotary Position Embeddings (RoPE)**: Instead of traditional positional embeddings, B6 uses RoPE for enhanced positional understanding and better generalization to different sequence lengths.

```python
def forward(self, x, seq_len=None):
    # Compute rotary embeddings dynamically
    if seq_len > self.max_seq_len_cached:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
```

- **Flexible Normalization Architecture**: 
  - Supports both Pre-LN and Post-LN configurations
  - Uses Layer Normalization with optional bias terms
  - Implements skip connections for improved gradient flow

```python
class FlashTransformerLayer(nn.Module):
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if not self.config.PRE_NORM:
            # Post-norm architecture
            attn_output = self.attention(x, attention_mask)
            x = x + attn_output
            x = self.attention_layer_norm(x)
        else:
            # Pre-norm architecture
            attn_output = self.attention(x, attention_mask)
            x = x + attn_output
```

- **Optimized Feed-Forward Network**:
  - Configurable hidden dimensions
  - GELU activation function
  - Dropout for regularization
  - Optional bias terms

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        self.fc1 = nn.Linear(config.EMBED_SIZE, config.HIDDEN_DIM,
                            bias=config.USE_BIAS_IN_FFN)
        self.fc2 = nn.Linear(config.HIDDEN_DIM, config.EMBED_SIZE,
                            bias=config.USE_BIAS_IN_FFN)
        self.activation = nn.GELU()
```

### Architecture Innovations

1. **Memory-Efficient Attention**:
   - Implements packed key-value cache for efficient memory usage
   - Uses optimized attention patterns for causal language modeling
   - Supports flexible attention masking patterns

2. **Gradient Flow Optimization**:
   - Carefully designed residual connections
   - Proper initialization schemes for stable training
   - Optional gradient checkpointing for memory efficiency

3. **Scalable Design**:
   - Hardware-aware configuration adaptation
   - Dynamic batch size handling
   - Automatic mixed precision support

The architecture is designed to scale efficiently across different GPU configurations while maintaining training stability and optimization effectiveness. The implementation supports both research experimentation and production deployment scenarios through its flexible configuration system.


## 📊 Training Features

- **Gradient checkpointing**: Enables Gradient checkpointing for reduced memory footprint
- **Mixed Precision Training**: Reduces memory usage and speeds up training
- **Learning Rate Scheduling**: Implements cosine annealing for better convergence
- **Validation-based Checkpointing**: Saves the best model based on validation loss

## 🔧 Customization

The model is highly configurable through the `Config` class:

- Dataset selection (`DATASET_NAME`)
- Model architecture (`NUM_LAYERS`, `NUM_HEADS`, `EMBED_SIZE`)
- Training parameters (`LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS`)
- Memory optimizations (`GRADIENT_ACCUMULATION_STEPS`)

## 📈 Monitoring

Training progress can be monitored through Weights & Biases (wandb):
- Loss tracking
- Learning rate scheduling
- Resource utilization
- Model checkpoints

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## 📝 License

This project is open-source and available under the Apache Version 2.0 License.

## 🙏 Acknowledgments

- The architecture is inspired by the transformer model design
- Uses HuggingFace's transformers library for tokenization
- Implements training optimizations from modern language model research

