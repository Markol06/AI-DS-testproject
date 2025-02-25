import sys
import os
import torch

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import the necessary modules
from NER.inference_ner import extract_animal_name
from Image_Classification.inference_image import load_model, predict

# 🚀 Determine execution device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Paths to trained models
IMAGE_MODEL_PATH = "D:/Projects/NER_Image_Classification/Image_Classification/model/image_classification.pth"
CLASS_NAMES_DIR = "D:/Projects/NER_Image_Classification/dataset/animals10/raw-img"

# Load class names
class_to_idx = {cls.lower(): i for i, cls in enumerate(os.listdir(CLASS_NAMES_DIR))}
idx_to_class = {i: cls.lower() for cls, i in class_to_idx.items()}
NUM_CLASSES = len(class_to_idx)

# Function to check if extracted animal matches the classified one
def is_match(text_animal, image_animal):
    return text_animal.lower() == image_animal.lower()

if __name__ == "__main__":
    # Get text and image input from user
    text_input = input("📜 Enter a description of the image (e.g., 'There is a cat in the picture.'): ").strip()
    image_path = input("🖼 Enter the path to the image: ").strip()

    # Check if the image exists
    if not os.path.exists(image_path):
        print("❌ Error: File not found! Please ensure the path is correct.")
        exit()

    # Extract animal name from text using the NER model
    text_animal = extract_animal_name(text_input)  # ❌ Removed model_path!
    if not text_animal:
        print("❌ No animal detected in the text! Please try again.")
        exit()

    print(f"🔍 Extracted animal from text: {text_animal}")

    # Load the image classification model
    model = load_model(IMAGE_MODEL_PATH)

    # Predict the animal in the image
    image_animal = predict(image_path, model)

    if image_animal == "undefined animal":
        print("⚠️ Warning: The animal in the image is not in the known classes.")
    else:
        print(f"🦁 Predicted animal from image: {image_animal}")

    # Compare results
    match_result = is_match(text_animal, image_animal)
    print(f"✅ Match result: {match_result}")