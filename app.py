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
    DATASET_SIZE = 5000000
    #DATASET_SIZE = 1000  # Very quick runs, basic testing; #DATASET_SIZE = 1000000 Medium Training Run; DATASET_SIZE = 8000000 or None  # Full dataset
    STREAM_BUFFER_SIZE = 10000  # Number of examples to buffer
    CACHE_DIR = "./dataset_cache"
    MAP_BATCH_SIZE = 1000
    
    # Dataset specific parameters
    DATASET_NAME = "c4"  # Change this to use different datasets
    TEXT_COLUMN = "text"  # Change based on dataset
    GRADIENT_ACCUMULATION_STEPS = 8

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

def train_model(model, train_loader, valid_loader, config, tokenizer):
    print("Initializing wandb...")
    # wandb.init(project="language-model-training")
    wandb.init(
        # set the wandb project where this run will be logged
        project="b6_large_language_model",

        # track hyperparameters and run metadata
        config={
            "learning_rate": config.LEARNING_RATE,
            "architecture": "Transformers",
            "dataset": "c4",
            "epochs": config.EPOCHS,
        }
    )
    accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS
    print("Setting up training...")
    optimizer = create_optimizer(model, config)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    best_valid_loss = float('inf')
    best_model_path = 'best_model.pt'
    
    print("Starting training...")
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for batch_idx, (input_ids, attention_mask) in enumerate(progress_bar):
            # Move tensors to device here
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, config.VOCAB_SIZE), 
                              shift_labels.view(-1))
                
            
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)            
            # scaler.step(optimizer)
            # scaler.update()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
            avg_loss = train_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        valid_loss = 0
        
        print("\nRunning validation...")
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(valid_loader):
                # Move tensors to device here
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss = criterion(shift_logits.view(-1, config.VOCAB_SIZE), 
                                  shift_labels.view(-1))
                valid_loss += loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "valid_loss": avg_valid_loss
        })
        
        print(f"\nEpoch {epoch+1}")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_valid_loss:.4f}")
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': best_valid_loss,
                'config': config,
            }, best_model_path)
            print(f"Saved new best model with validation loss: {best_valid_loss:.4f}")

def generate_text(model, prompt, tokenizer, config, max_length=100, temperature=0.7, top_k=50):
    model.eval()
    
    # Tokenize the prompt
    input_ids = tokenizer(
        prompt, return_tensors='pt')['input_ids'].to(device)
    attention_mask = torch.ones_like(input_ids)
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import math
import torch.nn.functional as F
import wandb
from torch.nn import LayerNorm
import numpy as np
from typing import Dict, List, Tuple
import os


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a proper Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]['text']
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )
        
        # Remove the batch dimension the tokenizer adds
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


# Modified collate function
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    # Don't move tensors to device here anymore
    return input_ids, attention_mask

def prepare_dataset_streaming(config, tokenizer):
    dataset = load_dataset(
        config.DATASET_NAME,
        streaming=True,
        split='train',
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR
    )
    
    def preprocess_function(examples):
        return tokenizer(
            examples[config.TEXT_COLUMN],
            truncation=True,
            max_length=config.SEQ_LENGTH,
            padding='max_length',
            return_tensors="pt"
        )
    
    # Process in batches
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=config.MAP_BATCH_SIZE,
        remove_columns=[config.TEXT_COLUMN]
    )
    
    return dataset

def load_mixed_datasets(config):
    datasets = []
    
    # Load multiple datasets
    for dataset_name in ["wikitext", "wikitext-103-v1"]:
        dataset = load_dataset(
            dataset_name,
            streaming=True,
            split='train',
            trust_remote_code=True,
            cache_dir=config.CACHE_DIR
        )
        datasets.append(dataset)
    
    # Interleave datasets
    def mixed_generator():
        iterators = [iter(dataset) for dataset in datasets]
        while True:
            for iterator in iterators:
                try:
                    yield next(iterator)
                except StopIteration:
                    break
    
    return mixed_generator()

def prepare_dataloaders(config):
    print("Loading and configuring tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # Choose your dataset configuration
    DATASET_CONFIG = {
        'openwebtext2': {
            'name': "the_pile_openwebtext2",
            'text_column': 'text',
        },
        'redpajama': {
            'name': "togethercomputer/RedPajama-Data-1T",
            'text_column': 'text',
        },
        'oscar': {
            'name': "oscar-corpus/OSCAR-2301",
            'text_column': 'text',
        },
        'stack': {
            'name': "bigcode/the-stack",
            'text_column': 'content',
        },
        'books3': {
            'name': "the_pile_books3",
            'text_column': 'text',
        },
        'openwebtext': {
            'name': "openwebtext",
            'text_column': 'text'
        },
        'c4': {
            'name': "allenai/c4",
            'text_column': 'text',
        }
    }
    
    # Select dataset configuration
    # dataset_config = DATASET_CONFIG['openwebtext2']  # Change this to use different datasets
    dataset_config = DATASET_CONFIG[config.DATASET_NAME]

    print(f"Loading {dataset_config['name']} dataset...")
    
    # Efficient streaming implementation
    dataset = load_dataset(
        dataset_config['name'], "en",
        streaming=True,
        split='train',
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR
    )
    
    def data_generator():
        for i, example in enumerate(dataset):
            if i >= config.DATASET_SIZE:
                break
            # Get text from correct column
            text = example[dataset_config['text_column']]
            yield {'text': text}
    
    dataset = list(data_generator())
    
    split_idx = int(0.9 * len(dataset))
    train_data = dataset[:split_idx]
    valid_data = dataset[split_idx:]
    
    train_dataset = TextDataset(train_data, tokenizer, config.SEQ_LENGTH)
    valid_dataset = TextDataset(valid_data, tokenizer, config.SEQ_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, valid_loader, tokenizer


def load_trained_model(model_path, config):
    """
    Load a trained model from a checkpoint file.
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path)
    model = ImprovedTransformerModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def run_text_generation_only(model_path, prompt):
    """
    Function to run text generation without training.
    Useful for generating text after the model has been trained.
    """
    config = Config()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the trained model
    model = load_trained_model(model_path, config)
    
    # Generate text
    generated_text = generate_text(
        model,
        prompt,
        tokenizer,
        config,
        max_length=150,
        temperature=0.7,
        top_k=50
    )
    
    return generated_text

# The rest of your model code remains the same...
# (ImprovedTransformerModel, MultiHeadAttention, FeedForward, TransformerLayer)
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # Initialize config
    config = Config()
    
    print(f"Using device: {device}")
    
    # Load data
    print("Preparing dataloaders...")
    train_loader, valid_loader, tokenizer = prepare_dataloaders(config)
    
    # Initialize model
    print("Initializing model...")
    model = ImprovedTransformerModel(config).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("Starting training...")
    train_model(model, train_loader, valid_loader, config, tokenizer)

def run_model():
    # Initialize config
    config = Config()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
     # Load the best model for text generation
    best_model_path = 'best_model.pt'
    if os.path.exists(best_model_path):
        model = load_trained_model(best_model_path, config)
        
        # Generate text examples
        test_prompts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "The most important scientific discovery"
        ]
        
        print("\nGenerating text samples:")
        for prompt in test_prompts:
            print("\nPrompt:", prompt)
            print("-" * 50)
            generated = generate_text(
                model,
                prompt,
                tokenizer,
                config,
                max_length=150,
                temperature=0.7,
                top_k=50
            )
            print("Generated text:", generated)
            print("-" * 50)
    else:
        print("\nNo trained model found. Please train the model first.")

if __name__ == "__main__":
    main()
    # run_model()
    # 0a1758d09b1514f46e5d0444193f4413a33f4f1e