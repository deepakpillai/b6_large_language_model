import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import wandb
from torch.nn import LayerNorm
import os
from typing import Dict, Optional, Tuple
import model as modelObj
from memorymanager import MemoryManager
from lr_scheduler import OptimizedLLMScheduler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainingManager:
    def __init__(self, config, early_stopping_patience: int = 3):
        self.config = config
        self.early_stopping_patience = early_stopping_patience
        self.best_valid_loss = float('inf')
        self.patience_counter = 0
        self.best_model_path = 'best_model.pt'
        self.checkpoint_path = 'checkpoint.pt'
        self.memory_manager = MemoryManager()
        
    def should_stop_early(self, valid_loss: float) -> bool:
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.early_stopping_patience
    
    def adjust_batch_size(self, loss_value: torch.Tensor) -> bool:
        if torch.isnan(loss_value) or torch.isinf(loss_value):
            self.config.BATCH_SIZE = max(1, self.config.BATCH_SIZE // 2)
            print(f"Reducing batch size to {self.config.BATCH_SIZE}")
            return True
        return False
    
    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: optim.lr_scheduler._LRScheduler, loss: float):
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, self.checkpoint_path)
    
    def save_best_model(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: optim.lr_scheduler._LRScheduler, valid_loss: float):
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_loss': valid_loss,
                'config': self.config,
            }, self.best_model_path)
            print(f"Saved new best model with validation loss: {valid_loss:.4f}")
    
    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer) -> int:
        start_epoch = 0
        if os.path.exists(self.checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        return start_epoch

def train_model(model, train_loader, valid_loader, config, tokenizer):
    print("Initializing training setup...")
    
    # Initialize training manager
    training_manager = TrainingManager(config)
    
    # Initialize wandb with error handling
    try:
        wandb.init(
            project="b6_large_language_model",
            config={
                "learning_rate": config.LEARNING_RATE,
                "architecture": "Transformers",
                "dataset": config.DATASET_NAME,
                "epochs": config.EPOCHS,
                "warmup_steps": config.WARMUP_STEPS,
                "min_lr_ratio": config.MIN_LR_RATIO,
                "warmup_init_lr": config.WARMUP_INIT_LR
            }
        )
    except Exception as e:
        print(f"Warning: wandb initialization failed: {e}")

    # Initialize training components
    optimizer = modelObj.create_optimizer(model, config)
    
    # Calculate total steps before creating scheduler
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    print(f"Total training steps: {total_steps}")
    
    # Initialize the optimized LLM scheduler
    scheduler = OptimizedLLMScheduler(
        optimizer=optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=config.WARMUP_STEPS,
        min_lr_ratio=config.MIN_LR_RATIO,
        warmup_init_lr=config.WARMUP_INIT_LR
    )
    
    # Load checkpoint if exists
    start_epoch = training_manager.load_checkpoint(model, optimizer)
    
    scaler = torch.amp.GradScaler(device='cuda')
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Initial memory cleanup
    training_manager.memory_manager.cleanup()
    
    print(f"Starting training with gradient accumulation steps: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate schedule: warmup steps={config.WARMUP_STEPS}, initial lr={config.WARMUP_INIT_LR:.2e}")
    
    step = 0  # Global step counter for learning rate scheduling
    
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for batch_idx, (input_ids, attention_mask) in enumerate(progress_bar):
            try:
                # Move tensors to device efficiently
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                
                # Automatic mixed precision training
                with torch.amp.autocast(device_type='cuda'):
                    try:
                        outputs = model(input_ids, attention_mask)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            training_manager.memory_manager.cleanup()
                            raise RuntimeError("GPU OOM - consider reducing batch size")
                        raise e
                    
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss = criterion(
                        shift_logits.view(-1, config.VOCAB_SIZE),
                        shift_labels.view(-1)
                    )
                    loss = loss / config.GRADIENT_ACCUMULATION_STEPS
                
                # Check for NaN/Inf loss and adjust batch size if needed
                if training_manager.adjust_batch_size(loss):
                    training_manager.memory_manager.cleanup()
                    continue
                
                # Gradient accumulation
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    # Unscale gradients for proper clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                    
                    # Optimizer and scheduler steps
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()  # Update learning rate
                    optimizer.zero_grad()
                    
                    step += 1  # Increment global step counter
                    
                    # Periodic memory cleanup
                    if batch_idx % (config.GRADIENT_ACCUMULATION_STEPS * 10) == 0:
                        training_manager.memory_manager.cleanup()
                
                train_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
                avg_loss = train_loss / (batch_idx + 1)
                current_lr = scheduler.get_lr()[0]  # Get current learning rate
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                # Log to wandb
                wandb.log({
                    "batch_loss": loss.item() * config.GRADIENT_ACCUMULATION_STEPS,
                    "learning_rate": current_lr,
                    "step": step
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    training_manager.memory_manager.cleanup()
                    if training_manager.adjust_batch_size(torch.tensor(float('inf'))):
                        continue
                raise e
        
        # Save checkpoint after each epoch
        training_manager.save_checkpoint(epoch, model, optimizer, scheduler, train_loss)
        
        # Validation phase
        model.eval()
        valid_loss = 0
        training_manager.memory_manager.cleanup()  # Clean memory before validation
        
        print("\nRunning validation...")
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(valid_loader):
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda'):
                    try:
                        outputs = model(input_ids, attention_mask)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            training_manager.memory_manager.cleanup()
                            raise RuntimeError("GPU OOM - consider reducing batch size")
                        raise e
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss = criterion(
                        shift_logits.view(-1, config.VOCAB_SIZE),
                        shift_labels.view(-1)
                    )
                valid_loss += loss.item()
                
                del outputs, shift_logits, shift_labels
        
        training_manager.memory_manager.cleanup()  # Clean memory after validation
        
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "valid_loss": avg_valid_loss,
            "learning_rate_epoch": current_lr
        })
        
        print(f"\nEpoch {epoch+1}")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_valid_loss:.4f}")
        print(f"Current learning rate: {current_lr:.2e}")
        
        # Save best model and check for early stopping
        training_manager.save_best_model(epoch, model, optimizer, scheduler, avg_valid_loss)
        
        if training_manager.should_stop_early(avg_valid_loss):
            print("Early stopping triggered")
            break

    return model