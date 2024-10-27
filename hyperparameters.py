# For different GPU memory sizes:

# VOCAB_SIZE = 50257  # Keep as is - this matches GPT-2's vocabulary size which is ideal
# EMBED_SIZE = 1024   # Increase to 1024 for better representation capacity
# NUM_HEADS = 16      # Increase to 16 (embed_size/64 is a common ratio)
# NUM_LAYERS = 24     # Increase to 24 for deeper network capacity
# HIDDEN_DIM = 4096   # Increase to 4096 (roughly 4x embed_size is common)
# BATCH_SIZE = 16     # Reduce from 32 to 16 to handle memory constraints
# SEQ_LENGTH = 512    # Increase to 512 for better context handling
# EPOCHS = 5          # Increase to 5 for better convergence
# LEARNING_RATE = 3e-4  # Slightly increase for faster initial learning
# WARMUP_STEPS = 2000   # Increase for more stable training
# DROPOUT = 0.1        # Keep as is - good balance
# GRADIENT_CLIP = 1.0   # Keep as is - good default
# NUM_WORKERS = 4       # Increase once debugging is complete

# 8GB GPU: Use BATCH_SIZE=8, SEQ_LENGTH=256
# 16GB GPU: Use the recommended values above
# 24GB+ GPU: Can increase BATCH_SIZE to 32 or SEQ_LENGTH to 1024

# Hyperparameters
class Config:
    VOCAB_SIZE = 50257 # Matches GPT-2's vocabulary size which is ideal
    EMBED_SIZE = 1024  #1024 for better representation capacity
    NUM_HEADS = 16    # (embed_size/64 is a common ratio)
    NUM_LAYERS = 24   # for deeper network capacity
    HIDDEN_DIM = 4096 # 4096 (roughly 4x embed_size is common)
    BATCH_SIZE = 12   # Reduce from 32 to 16 (for 16gb ram) to handle memory constraints
    SEQ_LENGTH = 256  # Increase to 512 for better context handling
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 2000 # Increase for more stable training
    DROPOUT = 0.1
    GRADIENT_CLIP = 1.0
    NUM_WORKERS = 0  # Changed to 0 initially to debug
    DATASET_SIZE = 20 #5000000
    #DATASET_SIZE = 1000  # Very quick runs, basic testing; #DATASET_SIZE = 1000000 Medium Training Run; DATASET_SIZE = 8000000 or None  # Full dataset
    STREAM_BUFFER_SIZE = 10000  # Number of examples to buffer
    CACHE_DIR = "./dataset_cache"
    MAP_BATCH_SIZE = 1000
    
    # Dataset specific parameters
    DATASET_NAME = "c4"  # Change this to use different datasets
    TEXT_COLUMN = "text"  # Change based on dataset
    GRADIENT_ACCUMULATION_STEPS = 8