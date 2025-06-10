import os
from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import torch.nn.functional as F
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the model architecture
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

class EfficientNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetWithAttention, self).__init__()
        weights = EfficientNet_B3_Weights.DEFAULT
        self.base_model = efficientnet_b3(weights=weights)
        self.spatial_attention = SpatialAttention()
        self.base_model.features[-1].add_module("spatial_attention", self.spatial_attention)
        self.num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        if not self.training:
            self.base_model.classifier[1].track_running_stats = False
        return self.base_model(x)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# First get the number of classes
num_classes = len([d for d in os.listdir('Dataset') if os.path.isdir(os.path.join('Dataset', d))])
# Load the complete model
model = torch.load('model.pth', map_location=device, weights_only=False)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get class names from the dataset folder
class_names = sorted(os.listdir('Dataset'))
severity_levels = ["Mild", "Moderate", "Severe"]

def predict_with_unknown(image_bytes, threshold=0.6):
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        
    if confidence.item() < threshold:
        return {"disease": "Unknown Category (Low Confidence)", "severity": None, "confidence": None}
    
    predicted_class_name = class_names[predicted_class.item()]
    
    # Predict severity
    class_index = predicted_class.item()
    severity_index = min(class_index // (len(class_names) // len(severity_levels)), len(severity_levels) - 1)
    severity = severity_levels[severity_index]
    
    return {
        "disease": predicted_class_name,
        "severity": severity,
        "confidence": round(confidence.item() * 100, 2)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        img_bytes = file.read()
        prediction = predict_with_unknown(img_bytes)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # For local development
    # app.run(debug=True)
    
    # For local network access (deployment)
    app.run(host='0.0.0.0', port=5000, debug=False)
