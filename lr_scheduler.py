import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class OptimizedLLMScheduler(LRScheduler):
    """
    Optimized learning rate scheduler for Large Language Models.
    Implements linear warmup followed by cosine decay.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = None,
        min_lr_ratio: float = 0.1,
        warmup_init_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps or min(10000, int(0.1 * num_training_steps))
        self.min_lr_ratio = min_lr_ratio
        self.warmup_init_lr = warmup_init_lr
        
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        
        super().__init__(optimizer, last_epoch)
    
    def _get_warmup_lr(self, step: int, base_lr: float) -> float:
        """Calculate learning rate during warmup phase"""
        lr_range = base_lr - self.warmup_init_lr
        progress = float(step) / float(max(1, self.num_warmup_steps))
        return self.warmup_init_lr + progress * lr_range
    
    def _get_cosine_lr(self, step: int, base_lr: float) -> float:
        """Calculate learning rate during cosine decay phase"""
        progress = float(step - self.num_warmup_steps) / \
                  float(max(1, self.num_training_steps - self.num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Minimum learning rate
        min_lr = base_lr * self.min_lr_ratio
        return min_lr + (base_lr - min_lr) * cosine_decay
    
    def get_lr(self) -> list:
        """Get current learning rate"""
        step = self.last_epoch
        
        if step < self.num_warmup_steps:
            # Linear warmup phase
            return [self._get_warmup_lr(step, base_lr) 
                   for base_lr in self.base_lrs]
        
        # Cosine decay phase
        return [self._get_cosine_lr(step, base_lr) 
                for base_lr in self.base_lrs]