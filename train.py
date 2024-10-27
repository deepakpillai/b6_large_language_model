
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
    model = modelObj.ImprovedTransformerModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def run_text_generation_only(model_path, prompt):
    """
    Function to run text generation without training.
    Useful for generating text after the model has been trained.
    """
    config = hyperparameters.Config()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the trained model
    model = load_trained_model(model_path, config)
    
    # Generate text
    generated_text = model.generate_text(
        model,
        prompt,
        tokenizer,
        config,
        max_length=150,
        temperature=0.7,
        top_k=50
    )
    
    return generated_text

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
    optimizer = modelObj.create_optimizer(model, config)
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
            
            with torch.autocast(device_type='cuda'):
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
                
                with torch.autocast(device_type='cuda'):
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