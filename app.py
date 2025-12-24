from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import io
import base64

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = ['Acne', 'Actinic_Keratosis', 'Bullous', 'Candidiasis', 'DrugEruption', 
               'Infestations_Bites', 'Lichen', 'Lupus', 'Moles', 'Psoriasis', 
               'Rosacea', 'Seborrh_Keratoses', 'SkinCancer', 'Sun_Sunlight_Damage', 
               'Tinea', 'Unknown', 'Vascular_Tumors', 'Vasculitis', 
               'Vitiligo', 'Warts']
NUM_CLASSES = len(class_names)

# Load model
model = EfficientNet.from_name("efficientnet-b0")
model._fc = torch.nn.Linear(model._fc.in_features, NUM_CLASSES)
state_dict = torch.load("CV_efficientnet_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", prediction="No image uploaded")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", prediction="No image selected")
    
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
    confidence = confidence.item() * 100

    # Convert image to Base64 for preview
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_data = f"data:image/png;base64,{img_str}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=f"{confidence:.2f}%",
        img_data=img_data
    )

if __name__ == "__main__":
    app.run(debug=True)
