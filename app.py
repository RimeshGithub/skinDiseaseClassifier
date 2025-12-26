from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import io
import base64
import pandas as pd
import json

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max upload

# Force CPU (Render has no GPU)
device = torch.device("cpu")

# Class names
class_names = [
    'Acne', 'Actinic_Keratosis', 'Bullous', 'Candidiasis', 'DrugEruption',
    'Infestations_Bites', 'Lichen', 'Lupus', 'Moles', 'Psoriasis',
    'Rosacea', 'Seborrh_Keratoses', 'SkinCancer', 'Sun_Sunlight_Damage',
    'Tinea', 'Unknown', 'Vascular_Tumors', 'Vasculitis',
    'Vitiligo', 'Warts'
]
NUM_CLASSES = len(class_names)

# ------------------------------
# Load disease information from CSV
# ------------------------------
try:
    disease_df = pd.read_csv("skin_diseases.csv")
    disease_info = {}
    for _, row in disease_df.iterrows():
        disease_info[row['Disease']] = {
            'description': row['Description'],
            'causes': row['Causes'],
            'precautions': [p.strip() for p in row['Precautions'].split(',')],
            'severity': row['Severity']
        }
except Exception as e:
    print(f"Error loading CSV: {e}")
    disease_info = {}

# ------------------------------
# Load model once at startup
# ------------------------------
model = EfficientNet.from_name("efficientnet-b0")
model._fc = torch.nn.Linear(model._fc.in_features, NUM_CLASSES)
state_dict = torch.load("efficientnet_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ------------------------------
# Image preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        # Read image in memory
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess and predict
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        prediction = class_names[pred.item()]
        confidence_value = confidence.item() * 100

        # Get disease information
        info = disease_info.get(prediction, {
            'description': 'No description available.',
            'causes': 'Causes not specified.',
            'precautions': ['Consult a dermatologist for proper diagnosis and treatment.'],
            'severity': 'Unknown'
        })

        # Convert image to Base64 for preview (JPEG, smaller memory)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data = f"data:image/jpeg;base64,{img_str}"

        # Return JSON response for AJAX or render template for form submission
        if request.headers.get('Accept') == 'application/json' or request.is_json:
            return jsonify({
                'prediction': prediction,
                'confidence': f"{confidence_value:.2f}%",
                'description': info['description'],
                'causes': info['causes'],
                'precautions': info['precautions'],
                'severity': info['severity'],
                'img_data': img_data
            })
        else:
            # For traditional form submission
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=f"{confidence_value:.2f}%",
                description=info['description'],
                causes=info['causes'],
                precautions_json=json.dumps(info['precautions']),
                severity=info['severity'],
                img_data=img_data
            )

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Run (for local testing)
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)