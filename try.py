import torch
import hyperparameters
from model import ImprovedTransformerModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = hyperparameters.setup_config(hyperparameters.Config(), "rtx3060")
print(config)
model = ImprovedTransformerModel(config).to(device)
input_ids = torch.randint(0, config.VOCAB_SIZE, (2, 512)).to(device)
attention_mask = torch.ones_like(input_ids)
outputs = model(input_ids, attention_mask)
print(outputs)