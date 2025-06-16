import torch
from app import EfficientNetWithAttention
import os

# Match number of classes from your project
num_classes = len([d for d in os.listdir('Dataset') if os.path.isdir(os.path.join('Dataset', d))])

# Load full model first (as you originally did)
model = torch.load('model.pth', map_location='cpu')

# Save only state_dict
torch.save(model.state_dict(), 'model_state_dict.pth')

print("✅ Saved state_dict as model_state_dict.pth")
