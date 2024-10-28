import torch
from dataclasses import dataclass
from typing import Dict, Any

# For different GPU memory sizes:

# VOCAB_SIZE = 50257  # Keep as is - this matches GPT-2's vocabulary size which is ideal
# EMBED_SIZE = 1024   # Increase to 2048 for better representation capacity
# NUM_HEADS = 16      # Increase to 16 (embed_size/64 is a common ratio)
# NUM_LAYERS = 24     # Increase to 32 for deeper network capacity
# HIDDEN_DIM = 4096   # Increase to 8192 (roughly 4x embed_size is common)
# BATCH_SIZE = 16     # Reduce from 32 to 16 to handle memory constraints
# SEQ_LENGTH = 2048    # Increase to 4096 for better context handling
# EPOCHS = 3          # Increase to 5 for better convergence
# LEARNING_RATE = 3e-4  # Slightly increase for faster initial learning
# WARMUP_STEPS = 2000   # Increase for more stable training
# DROPOUT = 0.1        # Keep as is - good balance
# GRADIENT_CLIP = 1.0   # Keep as is - good default
# NUM_WORKERS = 4       # Increase once debugging is complete

# 8GB GPU: Use BATCH_SIZE=8, SEQ_LENGTH=256
# 16GB GPU: Use the recommended values above
# 24GB+ GPU: Can increase BATCH_SIZE to 32 or SEQ_LENGTH to 4096

#The effective memory usage formula for transformer models is roughly:
#Memory ∝ BATCH_SIZE × SEQ_LENGTH × EMBED_SIZE × NUM_LAYERS
#Choose the configuration based on your priorities:
#If you need longer context understanding: Use the long-context config
#If you need better pattern recognition: Use the original optimized config with more layers
#If you need faster training: Use the optimized config with smaller sequence length
#Remember that the actual usable context length during inference can be different from training sequence length, 
#and you can potentially do inference on longer sequences than what you trained on (though with potentially degraded quality).

#Let's calculate the approximate memory requirements for this configuration:
#Memory ∝ BATCH_SIZE × SEQ_LENGTH × EMBED_SIZE × NUM_LAYERS
#Values:

#BATCH_SIZE = 16
#SEQ_LENGTH = 2048
#EMBED_SIZE = 1024
#NUM_LAYERS = 32

#Let's calculate step by step:

#Base Memory Calculation:
#16 × 2048 × 1024 × 32 = 1,073,741,824 parameters
#Converting to bytes:
#Each parameter typically requires 4 bytes for forward pass (float32)
#During training, we need:
#4 bytes for parameters
#4 bytes for gradients
#8 bytes for optimizer states (using Adam)
#Additional memory for activations and attention matrices

#Detailed Memory Calculation:
#Base memory: 1,073,741,824 × 4 = 4.29 GB (parameters)
#Gradients: 4.29 GB (same size as parameters)
#Optimizer states: 4.29 GB × 2 = 8.58 GB (Adam uses 2 states per parameter)
#Attention matrices: 16 × 2048 × 2048 × 32 × 4 = 8.59 GB
#Activations and temporary buffers: ≈ 4.29 GB
#Total estimated memory requirement:
#4.29 + 4.29 + 8.58 + 8.59 + 4.29 = 30.04 GB

#Additional considerations:
#PyTorch CUDA overhead: ~2-3 GB
#Memory fragmentation: ~10-15% overhead
#Gradient accumulation helps but you still need peak memory

#Final estimate: ~35-40 GB of VRAM required
#Recommendations for your setup:
#If you have a 40GB A100 GPU:
#This configuration should work, but it's cutting it close
#You might experience occasional OOM errors during training spikes


#Hyperparameters
class Config:
    # Model Architecture
    VOCAB_SIZE: int = 0
    EMBED_SIZE: int = 0
    NUM_HEADS: int = 0
    NUM_LAYERS: int = 0
    HIDDEN_DIM: int = 0
    MAX_POSITION_EMBEDDINGS: int = 0
    
    # Training Parameters
    BATCH_SIZE: int = 0
    SEQ_LENGTH: int = 0
    EPOCHS: int = 0
    LEARNING_RATE: float = 3e-4
    WARMUP_STEPS: int = 0
    DROPOUT: float = 0.1
    ATTENTION_DROPOUT: float = 0.1
    WEIGHT_DECAY: float = 0.1
    GRADIENT_CLIP: float = 1.0
    NUM_WORKERS: int = 0
    
    # Optimizer Parameters (for Lion)
    BETA1: float = 0.95
    BETA2: float = 0.98
    EPSILON: float = 1e-5
    
    # Memory Management
    GRADIENT_ACCUMULATION_STEPS: int = 0
    MIXED_PRECISION: bool = True
    USE_FLASH_ATTENTION: bool = True
    GRADIENT_CHECKPOINTING: bool = True
    
    # Flash Attention Specific
    FLASH_ATTENTION_VERSION: str = "2"
    PAD_TO_MULTIPLE_OF: int = 8
    
    # Architecture Specific
    USE_BIAS_IN_ATTN: bool = False
    USE_BIAS_IN_FFN: bool = True
    PRE_NORM: bool = True
    
    # Loss and Training
    LABEL_SMOOTHING: float = 0.0
    MAX_GRAD_NORM: float = 1.0
    
    # Dataset Parameters
    DATASET_SIZE: int = 0
    STREAM_BUFFER_SIZE: int = 0
    CACHE_DIR: str = "./dataset_cache"
    MAP_BATCH_SIZE: int = 0
    
    # Dataset Configuration
    DATASET_NAME: str = "c4"
    TEXT_COLUMN: str = "text"
    
    # Logging and Checkpointing
    LOG_INTERVAL: int = 0 
    SAVE_INTERVAL: int = 0
    EVAL_INTERVAL: int = 0


@dataclass
class RTX3060Values:
    # Model Architecture
    VOCAB_SIZE: int = 50257
    EMBED_SIZE: int = 768
    NUM_HEADS: int = 12
    NUM_LAYERS: int = 12
    HIDDEN_DIM: int = 3072
    MAX_POSITION_EMBEDDINGS: int = 2048  # Maximum sequence length for positional embeddings

    # Training Parameters
    BATCH_SIZE: int = 4
    SEQ_LENGTH: int = 1024
    EPOCHS: int = 3
    LEARNING_RATE: float = 3e-4
    WARMUP_STEPS: int = 750
    DROPOUT: float = 0.1
    GRADIENT_CLIP: float = 1.0
    NUM_WORKERS: int = 4
    ATTENTION_DROPOUT: float = 0.1        # Specific dropout for attention
    WEIGHT_DECAY: float = 0.1             # L2 regularization factor

    # Optimizer Parameters (for Lion)
    BETA1: float = 0.95                   # First momentum coefficient
    BETA2: float = 0.98                   # Second momentum coefficient
    EPSILON: float = 1e-5                 # Small constant for numerical stability

    # Memory Management
    GRADIENT_ACCUMULATION_STEPS: int = 16
    MIXED_PRECISION: bool = True
    USE_FLASH_ATTENTION: bool = True
    GRADIENT_CHECKPOINTING: bool = True    # Enable gradient checkpointing
    
    # Flash Attention Specific
    FLASH_ATTENTION_VERSION: str = "2"     # Using Flash Attention v2
    PAD_TO_MULTIPLE_OF: int = 8           # Pad sequence length to multiple of 8 for optimal performance
    
    # Architecture Specific
    USE_BIAS_IN_ATTN: bool = False        # Whether to use bias in attention projections
    USE_BIAS_IN_FFN: bool = True          # Whether to use bias in FFN
    PRE_NORM: bool = True                 # Whether to use Pre-LN or Post-LN
    
    # Loss and Training
    LABEL_SMOOTHING: float = 0.0          # Label smoothing factor
    MAX_GRAD_NORM: float = 1.0            # Maximum gradient norm for clipping

    # Dataset Parameters
    DATASET_SIZE: int = 50000
    STREAM_BUFFER_SIZE: int = 500000
    CACHE_DIR: str = "./dataset_cache"
    MAP_BATCH_SIZE: int = 64
    
    # Dataset Configuration
    DATASET_NAME: str = "c4"
    TEXT_COLUMN: str = "text"

    # Logging and Checkpointing
    LOG_INTERVAL: int = 100               # Steps between logging
    SAVE_INTERVAL: int = 1000             # Steps between model saves
    EVAL_INTERVAL: int = 500              # Steps between evaluations
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.EMBED_SIZE % self.NUM_HEADS == 0, \
            f"Embedding size {self.EMBED_SIZE} must be divisible by number of heads {self.NUM_HEADS}"
        assert self.SEQ_LENGTH <= self.MAX_POSITION_EMBEDDINGS, \
            f"Sequence length {self.SEQ_LENGTH} cannot exceed maximum position embeddings {self.MAX_POSITION_EMBEDDINGS}"
        assert self.HIDDEN_DIM >= self.EMBED_SIZE, \
            f"Hidden dimension {self.HIDDEN_DIM} should be >= embedding size {self.EMBED_SIZE}"
        
        # Compute derived values
        self.HEAD_DIM = self.EMBED_SIZE // self.NUM_HEADS
        assert self.HEAD_DIM % 8 == 0, \
            "Head dimension must be divisible by 8 for optimal Flash Attention performance"

class A100Values:
    # Model Architecture
    VOCAB_SIZE: int = 50257  # GPT-2 vocabulary size
    EMBED_SIZE: int = 1024   # Embedding dimension
    NUM_HEADS: int = 16      # Number of attention heads
    NUM_LAYERS: int = 24     # Number of transformer layers
    HIDDEN_DIM: int = 4096   # FFN hidden dimension
    MAX_POSITION_EMBEDDINGS: int = 2048  # Maximum sequence length for positional embeddings
    
    # Training Parameters
    BATCH_SIZE: int = 16
    SEQ_LENGTH: int = 2048
    EPOCHS: int = 5
    LEARNING_RATE: float = 3e-4
    WARMUP_STEPS: int = 1000
    DROPOUT: float = 0.1
    ATTENTION_DROPOUT: float = 0.1        # Specific dropout for attention
    WEIGHT_DECAY: float = 0.1             # L2 regularization factor
    GRADIENT_CLIP: float = 1.0
    NUM_WORKERS: int = 8
    
    # Optimizer Parameters (for Lion)
    BETA1: float = 0.95                   # First momentum coefficient
    BETA2: float = 0.98                   # Second momentum coefficient
    EPSILON: float = 1e-5                 # Small constant for numerical stability
    
    # Memory Management
    GRADIENT_ACCUMULATION_STEPS: int = 4
    MIXED_PRECISION: bool = True
    USE_FLASH_ATTENTION: bool = True
    GRADIENT_CHECKPOINTING: bool = True    # Enable gradient checkpointing
    
    # Flash Attention Specific
    FLASH_ATTENTION_VERSION: str = "2"     # Using Flash Attention v2
    PAD_TO_MULTIPLE_OF: int = 8           # Pad sequence length to multiple of 8 for optimal performance
    
    # Architecture Specific
    USE_BIAS_IN_ATTN: bool = False        # Whether to use bias in attention projections
    USE_BIAS_IN_FFN: bool = True          # Whether to use bias in FFN
    PRE_NORM: bool = True                 # Whether to use Pre-LN or Post-LN
    
    # Loss and Training
    LABEL_SMOOTHING: float = 0.0          # Label smoothing factor
    MAX_GRAD_NORM: float = 1.0            # Maximum gradient norm for clipping
    
    # Dataset Parameters
    DATASET_SIZE: int = 200000
    STREAM_BUFFER_SIZE: int = 1000000
    CACHE_DIR: str = "./dataset_cache"
    MAP_BATCH_SIZE: int = 128
    
    # Dataset Configuration
    DATASET_NAME: str = "c4"
    TEXT_COLUMN: str = "text"
    
    # Logging and Checkpointing
    LOG_INTERVAL: int = 100               # Steps between logging
    SAVE_INTERVAL: int = 1000             # Steps between model saves
    EVAL_INTERVAL: int = 500              # Steps between evaluations
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.EMBED_SIZE % self.NUM_HEADS == 0, \
            f"Embedding size {self.EMBED_SIZE} must be divisible by number of heads {self.NUM_HEADS}"
        assert self.SEQ_LENGTH <= self.MAX_POSITION_EMBEDDINGS, \
            f"Sequence length {self.SEQ_LENGTH} cannot exceed maximum position embeddings {self.MAX_POSITION_EMBEDDINGS}"
        assert self.HIDDEN_DIM >= self.EMBED_SIZE, \
            f"Hidden dimension {self.HIDDEN_DIM} should be >= embedding size {self.EMBED_SIZE}"
        
        # Compute derived values
        self.HEAD_DIM = self.EMBED_SIZE // self.NUM_HEADS
        assert self.HEAD_DIM % 8 == 0, \
            "Head dimension must be divisible by 8 for optimal Flash Attention performance"

def setup_config(config_class, hardware_type: str = "rtx3060"):
    """
    Sets up the Config class with values based on hardware type.
    
    Args:
        config_class: The Config class to be modified
        hardware_type: Either "rtx3060" or "a100"
    
    Returns:
        Modified Config class with appropriate values
    """
    # Select values based on hardware
    values = RTX3060Values() if hardware_type.lower() == "rtx3060" else A100Values()
    
    # Get all attributes from the values class
    config_values = {key: value for key, value in values.__dict__.items() 
                    if not key.startswith('_')}
    
    # Create a new class with updated values
    for key, value in config_values.items():
        setattr(config_class, key, value)
    
    # Validate the configuration
    total_memory_required = estimate_memory_requirements(config_class)
    available_memory = get_available_gpu_memory()
    
    print(f"Estimated memory required: {total_memory_required:.2f} GB")
    print(f"Available GPU memory: {available_memory:.2f} GB")
    
    if total_memory_required > available_memory:
        print("\nWARNING: Estimated memory requirement exceeds available GPU memory!")
        print("Consider adjusting batch size or model parameters.")
    
    return config_class

def estimate_memory_requirements(config) -> float:
    """
    Estimates the GPU memory requirements in GB for the given configuration.
    This is a rough estimation and actual usage might vary.
    """
    # Model parameters
    param_size = config.VOCAB_SIZE * config.EMBED_SIZE  # Embedding
    param_size += config.NUM_LAYERS * (
        config.EMBED_SIZE * config.HIDDEN_DIM * 2 +  # FFN
        config.EMBED_SIZE * config.EMBED_SIZE * 4    # Attention
    )
    
    # Convert to GB and account for optimizer states
    param_memory = param_size * 4 * 4 / (1024**3)  # 4 bytes per parameter * 4 for optimizer states
    
    # Activation memory (rough estimation)
    batch_memory = (config.BATCH_SIZE * config.SEQ_LENGTH * config.EMBED_SIZE * 
                   config.NUM_LAYERS * 4) / (1024**3)
    
    # Add some overhead for CUDA kernels and other allocations
    total_memory = param_memory + batch_memory + 2  # 2GB overhead
    
    return total_memory

def get_available_gpu_memory() -> float:
    """
    Returns available GPU memory in GB.
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(device)
        return prop.total_memory / (1024**3)
    return 0.0