import os
from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import torch.nn.functional as F
import io

from model_arch import EfficientNetWithAttention  # ✅ Moved architecture to separate file

# ----------------- Flask Setup -----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Load Classes -----------------
DATASET_DIR = 'Dataset'
class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
num_classes = len(class_names)

# ----------------- Load Model -----------------
model = EfficientNetWithAttention(num_classes)
model.load_state_dict(torch.load('model_state_dict.pth', map_location=device))
model.to(device)
model.eval()

# ----------------- Image Transform -----------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

severity_levels = ["Mild", "Moderate", "Severe"]

# ----------------- Prediction Function -----------------
def predict_with_unknown(image_bytes, threshold=0.6):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    if confidence.item() < threshold:
        return {"disease": "Unknown Category (Low Confidence)", "severity": None, "confidence": None}

    predicted_class_name = class_names[predicted_class.item()]
    severity_index = min(predicted_class.item() // (len(class_names) // len(severity_levels)), len(severity_levels) - 1)
    severity = severity_levels[severity_index]

    return {
        "disease": predicted_class_name,
        "severity": severity,
        "confidence": round(confidence.item() * 100, 2)
    }

# ----------------- Routes -----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    try:
        img_bytes = file.read()
        prediction = predict_with_unknown(img_bytes)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)})

# ----------------- Run Server -----------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
