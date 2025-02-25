import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from glob import glob
from tqdm import tqdm
import time
from datetime import timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 🚀 Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

# Paths
image_dataset_path = "D:/Projects/NER_Image_Classification/dataset/animals10/raw-img"

# Image settings
IMG_SIZE = 224
BATCH_SIZE = 8  # Reduce batch size for debugging
EPOCHS = 10
NUM_CLASSES = len(os.listdir(image_dataset_path))

# Collect image paths
image_formats = ['*.jpg', '*.jpeg', '*.png']
image_paths = []
labels = []
class_to_idx = {cls: i for i, cls in enumerate(os.listdir(image_dataset_path))}

print("🔍 Collecting image paths and labels...")
start_time = time.time()
for cls in os.listdir(image_dataset_path):
    class_path = os.path.join(image_dataset_path, cls)
    images = []
    for fmt in image_formats:
        images.extend(glob(os.path.join(class_path, fmt)))
    for img in images:
        image_paths.append(img)
        labels.append(class_to_idx[cls])

print(f"✅ Collected {len(image_paths)} images from {NUM_CLASSES} classes in {time.time() - start_time:.2f} sec.")

# Check for grayscale images
def check_grayscale(image_path):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    return np.all(img_array[:, :, 0] == img_array[:, :, 1])

grayscale_images = [img for img in image_paths if check_grayscale(img)]
print(f"🔍 Found {len(grayscale_images)} grayscale images. Converting to RGB...")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")  # Ensure RGB
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def evaluate_model(loader, model, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy, precision, recall, f1



if __name__ == "__main__":
    print("📌 Creating dataset...")
    dataset = ImageDataset(image_paths, labels, transform=transform)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print(f"📊 Data split: Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Test single batch loading
    print("🔍 Testing single batch loading to detect issues...")
    test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    sample_batch = next(iter(test_loader))
    print(f"✅ Batch loaded successfully with shape {sample_batch[0].shape}")

    # Use DataLoader with reduced num_workers for debugging
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("🔍 Loading pre-trained ResNet model...")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 🚀 Start the training loop with CUDA support
    print("🚀 Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()  # Track epoch start time

        # Initialize tqdm progress bar for the entire epoch
        progress_bar = tqdm(train_loader, desc=f"📊 Epoch {epoch + 1}/{EPOCHS}", leave=True)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device, non_blocking=True), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update tqdm bar with current loss and estimated batch time
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Batch Time": f"{progress_bar.format_dict['elapsed'] / (batch_idx + 1):.3f}s"
            })
            progress_bar.update(1)  # Ensure progress bar updates properly

            # Log every 100 batches for better tracking
            if batch_idx % 500 == 0:
                elapsed_time = time.time() - start_time
                print(f"🟢 Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Elapsed Time: {timedelta(seconds=int(elapsed_time))}")

        # Compute average loss and total time for the epoch
        avg_loss = running_loss / len(train_loader)
        total_time = time.time() - start_time
        print(f"📈 Epoch {epoch + 1}/{EPOCHS} Complete | Avg Loss: {avg_loss:.4f} | "
              f"Total Time: {timedelta(seconds=int(total_time))}")
        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate_model(val_loader, model, criterion)
        print(f"📊 Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, "
              f"Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")

    print("✅ Training complete!")

    test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate_model(test_loader, model, criterion)
    print(f"🎯 Final Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, "
          f"Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}")

    # Save model
    model_save_path = "D:/Projects/NER_Image_Classification/Image_Classification/model/image_classification.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")

