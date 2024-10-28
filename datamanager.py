
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
import model as modelObj
import hyperparameters

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
    ).shuffle(buffer_size=config.STREAM_BUFFER_SIZE)
    
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
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    # Select dataset configuration
    dataset_config = config.DATASET_CONFIG.get(config.DATASET_NAME)
    if not dataset_config:
        raise ValueError(f"Unsupported dataset: {config.DATASET_NAME}")

    print(f"Loading {dataset_config['name']} dataset...")
    
    try:
        # Implement true streaming without loading entire dataset into memory
        dataset = load_dataset(
            dataset_config['name'],
            "en",
            streaming=True,
            split='train',
            trust_remote_code=True,
            cache_dir=config.CACHE_DIR
        )
        
        # Create efficient data iterator
        train_dataset = dataset.take(int(config.DATASET_SIZE * 0.9))
        valid_dataset = dataset.skip(int(config.DATASET_SIZE * 0.9)).take(int(config.DATASET_SIZE * 0.1))
        
        def create_dataloader(dataset, is_train=True):
            return DataLoader(
                TextDataset(dataset, tokenizer, config.SEQ_LENGTH),
                batch_size=config.BATCH_SIZE,
                shuffle=is_train,
                num_workers=config.NUM_WORKERS,
                collate_fn=collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            
        train_loader = create_dataloader(train_dataset, is_train=True)
        valid_loader = create_dataloader(valid_dataset, is_train=False)
        
        return train_loader, valid_loader, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def load_trained_model(model_path, config):
    """
    Load a trained model from a checkpoint file.
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path)
    model = modelObj.ImprovedTransformerModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


# # Optimized Training Function to run on a i3 12100, RTX 3060 and 36 GB RAM
# def train_model(model, train_loader, valid_loader, config, tokenizer):
#     print("Initializing training setup...")
    
#     # Initialize wandb with error handling
#     try:
#         wandb.init(
#             project="b6_large_language_model",
#             config={
#                 "learning_rate": config.LEARNING_RATE,
#                 "architecture": "Transformers",
#                 "dataset": config.DATASET_NAME,
#                 "epochs": config.EPOCHS,
#             }
#         )
#     except Exception as e:
#         print(f"Warning: wandb initialization failed: {e}")

#     # Initialize training components
#     optimizer = modelObj.create_optimizer(model, config)
    
#     # Add checkpointing for training interruption
#     start_epoch = 0
#     if os.path.exists('checkpoint.pt'):
#         print("Loading checkpoint...")
#         checkpoint = torch.load('checkpoint.pt')
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         start_epoch = checkpoint['epoch']
#         print(f"Resuming from epoch {start_epoch}")
    
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr=config.LEARNING_RATE,
#         epochs=config.EPOCHS,
#         steps_per_epoch=len(train_loader),
#         pct_start=0.05
#     )
    
#     scaler = torch.amp.GradScaler(device='cuda')
#     criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
#     # Improved memory management function
#     def cleanup_memory():
#         torch.cuda.empty_cache()
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
    
#     # Dynamic batch size adjustment
#     def adjust_batch_size(loss_value):
#         if torch.isnan(loss_value) or torch.isinf(loss_value):
#             config.BATCH_SIZE = max(1, config.BATCH_SIZE // 2)
#             print(f"Reducing batch size to {config.BATCH_SIZE}")
#             return True
#         return False
    
#     # Initial memory cleanup
#     cleanup_memory()
    
#     best_valid_loss = float('inf')
#     best_model_path = 'best_model.pt'
    
#     print(f"Starting training with gradient accumulation steps: {config.GRADIENT_ACCUMULATION_STEPS}")
    
#     for epoch in range(start_epoch, config.EPOCHS):
#         model.train()
#         train_loss = 0
#         optimizer.zero_grad()
        
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
#         for batch_idx, (input_ids, attention_mask) in enumerate(progress_bar):
#             try:
#                 # Move tensors to device efficiently
#                 input_ids = input_ids.to(device, non_blocking=True)
#                 attention_mask = attention_mask.to(device, non_blocking=True)
                
#                 # Automatic mixed precision training
#                 with torch.amp.autocast(device_type='cuda'):
#                     # outputs = model(input_ids, attention_mask)
#                     try:
#                         outputs = model(input_ids, attention_mask)
#                     except RuntimeError as e:
#                         if "out of memory" in str(e):
#                             torch.cuda.empty_cache()
#                             # Implement proper recovery strategy
#                             raise RuntimeError("GPU OOM - consider reducing batch size")
#                         raise e
                    
#                     shift_logits = outputs[..., :-1, :].contiguous()
#                     shift_labels = input_ids[..., 1:].contiguous()
#                     loss = criterion(
#                         shift_logits.view(-1, config.VOCAB_SIZE),
#                         shift_labels.view(-1)
#                     )
#                     loss = loss / config.GRADIENT_ACCUMULATION_STEPS
                
#                 # Check for NaN/Inf loss and adjust batch size if needed
#                 if adjust_batch_size(loss):
#                     cleanup_memory()
#                     continue
                
#                 # Gradient accumulation
#                 scaler.scale(loss).backward()
                
#                 if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
#                     scaler.step(optimizer)
#                     scaler.update()
#                     scheduler.step()
#                     optimizer.zero_grad()
                    
#                     # Periodic memory cleanup
#                     if batch_idx % (config.GRADIENT_ACCUMULATION_STEPS * 10) == 0:
#                         cleanup_memory()
                
#                 train_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
#                 avg_loss = train_loss / (batch_idx + 1)
                
#                 # Update progress bar
#                 progress_bar.set_postfix({
#                     'loss': f'{avg_loss:.4f}',
#                     'lr': f'{scheduler.get_last_lr()[0]:.2e}'
#                 })
                
#                 # Log to wandb
#                 wandb.log({
#                     "batch_loss": loss.item() * config.GRADIENT_ACCUMULATION_STEPS,
#                     "learning_rate": scheduler.get_last_lr()[0]
#                 })
                
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     cleanup_memory()
#                     if adjust_batch_size(torch.tensor(float('inf'))):
#                         continue
#                 raise e
        
#         # Save checkpoint after each epoch
#         torch.save({
#             'epoch': epoch + 1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'loss': train_loss,
#         }, 'checkpoint.pt')
        
#         # Validation phase
#         model.eval()
#         valid_loss = 0
#         cleanup_memory()  # Clean memory before validation
        
#         print("\nRunning validation...")
#         with torch.no_grad():
#             for input_ids, attention_mask in tqdm(valid_loader):
#                 input_ids = input_ids.to(device, non_blocking=True)
#                 attention_mask = attention_mask.to(device, non_blocking=True)
                
#                 with torch.amp.autocast(device_type='cuda'):
#                     # outputs = model(input_ids, attention_mask)
#                     try:
#                         outputs = model(input_ids, attention_mask)
#                     except RuntimeError as e:
#                         if "out of memory" in str(e):
#                             torch.cuda.empty_cache()
#                             # Implement proper recovery strategy
#                             raise RuntimeError("GPU OOM - consider reducing batch size")
#                         raise e
#                     shift_logits = outputs[..., :-1, :].contiguous()
#                     shift_labels = input_ids[..., 1:].contiguous()
#                     loss = criterion(
#                         shift_logits.view(-1, config.VOCAB_SIZE),
#                         shift_labels.view(-1)
#                     )
#                 valid_loss += loss.item()
                
#                 del outputs, shift_logits, shift_labels
        
#         cleanup_memory()  # Clean memory after validation
        
#         avg_train_loss = train_loss / len(train_loader)
#         avg_valid_loss = valid_loss / len(valid_loader)
        
#         wandb.log({
#             "epoch": epoch,
#             "train_loss": avg_train_loss,
#             "valid_loss": avg_valid_loss
#         })
        
#         print(f"\nEpoch {epoch+1}")
#         print(f"Average training loss: {avg_train_loss:.4f}")
#         print(f"Average validation loss: {avg_valid_loss:.4f}")
        
#         # Save best model
#         if avg_valid_loss < best_valid_loss:
#             best_valid_loss = avg_valid_loss
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'valid_loss': best_valid_loss,
#                 'config': config,
#             }, best_model_path)
#             print(f"Saved new best model with validation loss: {best_valid_loss:.4f}")