import train
import torch
import numpy as np
from transformers import GPT2TokenizerFast
import os
import hyperparameters
import model as modelObj
import inferencing

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def start_training():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # Initialize config
    # For RTX 3060 setup
    config = hyperparameters.setup_config(hyperparameters.Config(), "rtx3060")
    
    print(f"Using device: {device}")
    
    # Load data
    print("Preparing dataloaders...")
    train_loader, valid_loader, tokenizer = train.prepare_dataloaders(config)
    
    # Initialize model
    print("Initializing model...")
    model = modelObj.ImprovedTransformerModel(config).to(device)
    # Enable gradient checkpointing
    model.enable_gradient_checkpointing()

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("Starting training...")
    train.train_model(model, train_loader, valid_loader, config, tokenizer)

def run_model():
    # Initialize config
    config = hyperparameters.setup_config(hyperparameters.Config(), "rtx3060")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
     # Load the best model for text generation
    best_model_path = 'best_model.pt'
    if os.path.exists(best_model_path):
        model = train.load_trained_model(best_model_path, config)
        
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
            generated = inferencing.generate_text(
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
    start_training()
    # run_model()