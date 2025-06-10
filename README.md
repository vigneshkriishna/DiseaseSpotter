### 🌿 DiseaseSpotter: Real-Time Plant Disease Detection with EfficientNet-B3 & Spatial Attention

🚀 An AI-powered web app that instantly detects plant leaf diseases and offers actionable treatment tips. Built using the PlantDoc dataset and enhanced with EfficientNet-B3 + Spatial Attention for next-gen accuracy.

📸 Demo Preview
1️⃣ Upload a Plant Leaf Image
Drag & drop or upload leaf images in JPG/PNG formats.
![Screenshot 2025-06-10 024313](https://github.com/user-attachments/assets/53376eb4-42d9-46a8-8b24-a5131869db3d)


2️⃣ Click 'Analyze'
Get real-time AI-driven diagnosis powered by EfficientNet-B3.
![Screenshot 2025-06-10 024655](https://github.com/user-attachments/assets/b9fd4228-6641-400c-b6f6-dd98c37c8e93)


3️⃣ Get Instant Results
💡 Disease name, 🧪 severity level, ✅ treatment suggestions, and 🛡️ prevention tips.
![Screenshot 2025-06-10 024717](https://github.com/user-attachments/assets/b777fdcf-23ea-4125-a205-38fccf79a51f)

## ✨ Features
⚡️ EfficientNet-B3 Backbone: Lightweight yet powerful CNN model.

🧠 Spatial Attention Module: Automatically zooms in on diseased regions of the leaf.

🧪 Real-Time Inference: Optimized for instant predictions.

🌿 PlantDoc Dataset: Real-world, diverse disease dataset.

🔁 Transfer Learning: Pretrained model fine-tuned for this specific use-case.

🌐 Flask Web App: Simple, interactive frontend to upload and analyze images.

## 📚 Dataset: PlantDoc
📎 PlantDoc GitHub

~2,600 labeled images of healthy/diseased leaves.

Classes: Apple, Potato, Tomato, Bell Pepper, etc.

Suitable for multi-class supervised classification.

## 🧠 Model Architecture
EfficientNet-B3 → Spatial Attention Layer → Fully Connected Layer → Softmax
- Feature Extractor: EfficientNet-B3
- Custom Enhancement: Spatial Attention (focuses on disease spots)
- Classifier: Fully connected + softmax for prediction

 ## 🧪 Evaluation Metrics
  | Metric    | Score  |
| --------- | ------ |
| Accuracy  | 72.53% |
| Precision | 72.98% |
| Recall    | 72.53% |
| F1-Score  | 71.02% |

## 🚀 How to Run
1️⃣ Clone & Setup

git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt

2️⃣ Prepare Dataset
Download PlantDoc Dataset and structure it like:
plant-disease-detection/
└── data/
    └── plantdoc/
        ├── train/
        └── test/

🏋️‍♂️ Training
python train.py --data_dir data/plantdoc --model efficientnet_b3 --use_attention

🔍 Inference (Test Image)
python inference.py --image_path samples/test_image.jpg

## 🌐 Launch Flask Web App
python app.py
Go to: http://127.0.0.1:5000

## 🧠 App Capabilities
📷 Upload plant leaf images from your device

🔍 Real-time prediction with high accuracy

📋 Instant display of disease class + treatment tips

## 🔧 Dependencies
Python 3.8+

PyTorch ≥ 1.10

torchvision

efficientnet-pytorch

Flask

numpy, pandas, matplotlib

## Install all at once:

Copy code
pip install -r requirements.txt

🗂️ Project Structure
<pre>
plant-disease-detection/
├── app.py                  # Flask Web App
├── train.py                # Model training
├── inference.py            # Single image prediction
├── requirements.txt
├── data/                   # Dataset directory
│   └── plantdoc/
│       ├── train/
│       └── test/
├── models/
│   └── efficientnet_b3_attention.py
└── README.md
</pre>


🙌 Acknowledgements
📚 PlantDoc Dataset Contributors

📊 EfficientNet Researchers (Google AI)

🧠 Open-source ML & PyTorch community

📬 Contact
For bugs, ideas, collabs — open an issue or connect via GitHub.


