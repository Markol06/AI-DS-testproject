import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# 🚀 Determine the execution device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Path to the saved model
MODEL_PATH = "D:/Projects/NER_Image_Classification/Image_Classification/model/image_classification.pth"
CLASS_NAMES_DIR = "D:/Projects/NER_Image_Classification/dataset/animals10/raw-img"

# Load class names
class_to_idx = {cls: i for i, cls in enumerate(os.listdir(CLASS_NAMES_DIR))}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}
NUM_CLASSES = len(class_to_idx)


# Function to load the model
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Function to predict the class of an image
def predict(image_path, model, confidence_threshold=0.7):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
        max_prob, predicted_idx = torch.max(probabilities, 1)

    predicted_class = idx_to_class[predicted_idx.item()]

    # If the model is not confident enough, return "undefined animal"
    if max_prob.item() < confidence_threshold:
        return "undefined animal"

    return predicted_class


if __name__ == "__main__":
    # Get image path from input()
    image_path = input("🖼 Enter the path to the image: ").strip()

    if not os.path.exists(image_path):
        print("❌ Error: File not found! Please ensure the path is correct.")
    else:
        # Load the model
        model = load_model(MODEL_PATH)

        # Perform prediction
        predicted_label = predict(image_path, model)
        print(f"🦁 Predicted animal class: {predicted_label}")
