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

# Hyperparameters
# class Config:
#     VOCAB_SIZE = 50257 # Matches GPT-2's vocabulary size which is ideal
#     EMBED_SIZE = 1024  #1024 for better representation capacity
#     NUM_HEADS = 16    # (embed_size/64 is a common ratio)
#     NUM_LAYERS = 24   # for deeper network capacity
#     HIDDEN_DIM = 4096 # 4096 (roughly 4x embed_size is common)
#     BATCH_SIZE = 12   # Reduce from 32 to 16 (for 16gb ram) to handle memory constraints
#     SEQ_LENGTH = 256  # Increase to 512 for better context handling
#     EPOCHS = 5
#     LEARNING_RATE = 1e-4
#     WARMUP_STEPS = 2000 # Increase for more stable training
#     DROPOUT = 0.1
#     GRADIENT_CLIP = 1.0
#     NUM_WORKERS = 0  # Changed to 0 initially to debug
#     DATASET_SIZE = 100000
#     #DATASET_SIZE = 1000  # Very quick runs, basic testing; #DATASET_SIZE = 1000000 for Medium Training Run; DATASET_SIZE = 8000000 or None for Full dataset
#     STREAM_BUFFER_SIZE = 10000  # Number of examples to buffer
#     CACHE_DIR = "./dataset_cache"
#     MAP_BATCH_SIZE = 1000
    
#     # Dataset specific parameters
#     DATASET_NAME = "c4"  # Change this to use different datasets
#     TEXT_COLUMN = "text"  # Change based on dataset
#     GRADIENT_ACCUMULATION_STEPS = 8

# Optimized Hyperparameters for i3 12100 & RTX 3060
class Config:
    # Model Architecture - Optimized for 12GB VRAM
    VOCAB_SIZE = 50257  # Keep as is - matches GPT-2's vocabulary
    EMBED_SIZE = 768    # Reduced from 1024 to save memory
    NUM_HEADS = 12      # Reduced from 16 (embed_size/64 ratio)
    NUM_LAYERS = 12     # Reduced from 24 for memory efficiency
    HIDDEN_DIM = 3072   # Reduced from 4096 (4x embed_size)
    
    # Training Parameters
    BATCH_SIZE = 4      # Reduced to prevent OOM
    SEQ_LENGTH = 1024   # Reduced from 4096 for memory efficiency
    EPOCHS = 3
    LEARNING_RATE = 5e-5  # Slightly reduced for stability
    WARMUP_STEPS = 1000   # Reduced from 2000
    DROPOUT = 0.1
    GRADIENT_CLIP = 1.0
    NUM_WORKERS = 2      # Balanced for i3 12100 (4 cores)
    
    # Memory Management
    GRADIENT_ACCUMULATION_STEPS = 16  # Increased to simulate larger batch size
    MIXED_PRECISION = True  # Enable automatic mixed precision
    
    # Dataset Parameters
    DATASET_SIZE = 100000  # Start with smaller dataset for testing
    STREAM_BUFFER_SIZE = 1000000  # Reduced buffer size
    CACHE_DIR = "./dataset_cache"
    MAP_BATCH_SIZE = 100  # Reduced from 1000
    
    # Dataset Configuration
    DATASET_NAME = "c4"
    TEXT_COLUMN = "text"