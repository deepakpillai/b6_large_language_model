
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
    try:
        dataset = load_dataset(
            config.DATASET_NAME,
            streaming=True,
            split='train',
            trust_remote_code=True,
            cache_dir=config.CACHE_DIR
        ).shuffle(buffer_size=config.STREAM_BUFFER_SIZE)
        if not dataset:
            raise ValueError("Empty dataset loaded")
            return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}")
    
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
