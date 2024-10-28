import torch

class MemoryManager:
    @staticmethod
    def cleanup():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_stats():
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,
                'cached': torch.cuda.memory_reserved() / 1024**2
            }
        return {'allocated': 0, 'cached': 0}