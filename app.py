import json
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/vehicle_type_resnet18.pth"
CLASS_PATH = "class_names.json"

@st.cache_resource
def load_model():
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, len(class_names))
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    )

    model.eval()
    return model, class_names


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_vehicle(image, model, class_names):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)

    vehicle_type = class_names[predicted_index.item()]
    confidence_score = confidence.item() * 100

    return vehicle_type, confidence_score


def assign_parking(vehicle_type):
    parking_rules = {
        "Hatchback": "Compact Parking Zone",
        "Sedan": "Standard Parking Zone",
        "SUV": "Large Vehicle Zone",
        "Pickup": "Pickup / Truck Zone",
        "Other": "Manual Review Zone"
    }

    return parking_rules.get(vehicle_type, "Manual Review Zone")


st.set_page_config(
    page_title="Smart Valet Vehicle Classifier",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Smart Valet Vehicle Classifier")
st.write("Upload a vehicle image and the ML model will predict the vehicle type.")

model, class_names = load_model()

uploaded_file = st.file_uploader(
    "Upload vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Vehicle Image", use_container_width=True)

    vehicle_type, confidence = predict_vehicle(image, model, class_names)
    parking_zone = assign_parking(vehicle_type)

    st.subheader("Prediction Result")
    st.success(f"Vehicle Type: {vehicle_type}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Recommended Parking Zone: {parking_zone}")

    st.divider()

    st.write("### Parking Logic")
    st.write({
        "Detected Vehicle Type": vehicle_type,
        "Assigned Parking Zone": parking_zone
    })
