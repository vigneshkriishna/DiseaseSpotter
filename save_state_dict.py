import torch
import os
from model_arch import EfficientNetWithAttention, SpatialAttention
import torch.serialization

# Allowlist the custom classes
torch.serialization.add_safe_globals({
    'EfficientNetWithAttention': EfficientNetWithAttention,
    'SpatialAttention': SpatialAttention
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Infer class count
num_classes = len([d for d in os.listdir('Dataset') if os.path.isdir(os.path.join('Dataset', d))])

# Load full model
model = torch.load("model.pth", map_location=device, weights_only=False)

# Save the weights only
torch.save(model.state_dict(), "model_state_dict.pth")
print("✅ model_state_dict.pth saved successfully!")
