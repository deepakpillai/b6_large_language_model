# B6 Language Model 🤖

B6 is a next-gen, transformer-based language model built to bring high scalability and flexibility to NLP tasks, powered by PyTorch. Designed for peak performance, B6 merges modern architecture with cutting-edge training enhancements, making it a powerful tool for both research and practical applications in natural language processing.

## ✨ Key Features

- **Scalable Architecture**
  - Configurable model dimensions (embedding size, number of heads, layers)
  - Memory-efficient implementation with support for different GPU sizes
  - Gradient accumulation for handling larger batch sizes
  - Mixed-precision training with automatic mixed precision (AMP)

- **Advanced Training Components**
  - Multi-head attention mechanism with optional attention masking
  - Position embeddings for sequence understanding
  - Layer normalization and residual connections
  - GELU activation functions
  - AdamW optimizer with weight decay fix
  - Cosine learning rate scheduling

- **Optimized Data Pipeline**
  - Efficient data streaming implementation
  - Support for multiple datasets (C4, OpenWebText, RedPajama, OSCAR, etc.)
  - Custom data collation and preprocessing
  - Configurable sequence lengths and batch sizes

- **Modern Training Features**
  - Wandb integration for experiment tracking
  - Checkpoint saving and loading
  - Validation-based model selection
  - Gradient clipping for stability
  - Dropout for regularization

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers datasets wandb tqdm numpy zstandard
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

### GPU Memory Requirements

The model can be configured for different GPU memory sizes:
- 8GB GPU: Use `BATCH_SIZE=8`, `SEQ_LENGTH=256`
- 16GB GPU: Use default values
- 24GB+ GPU: Can increase `BATCH_SIZE` to 32 or `SEQ_LENGTH` to 4096

## 💡 Model Architecture

B6 implements a transformer architecture with several modern improvements:

- **Multi-head Self-attention**: Allows the model to attend to different parts of the input sequence
- **Feed-forward Networks**: Processes the attention output through position-wise fully connected layers
- **Layer Normalization**: Stabilizes training by normalizing activations
- **Residual Connections**: Helps with gradient flow in deep networks

## 📊 Training Features

- **Gradient Accumulation**: Enables training with larger effective batch sizes
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

