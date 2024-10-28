import torch
import torch.nn.functional as F

def generate_text(model, prompt, tokenizer, config, max_length=100, temperature=0.7, top_k=50):
    model.eval()
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Tokenize the prompt
    input_ids = tokenizer(
        prompt, return_tensors='pt')['input_ids'].to(device)
    attention_mask = torch.ones_like(input_ids)
    
    for _ in range(max_length):
        with torch.no_grad():
            # outputs = model(input_ids, attention_mask)
            try:
                outputs = model(input_ids, attention_mask)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    # Implement proper recovery strategy
                    raise RuntimeError("GPU OOM - consider reducing batch size")
                raise e
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