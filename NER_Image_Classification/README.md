Named Entity Recognition + Image Classification Pipeline
📌 Project Overview

This project integrates Named Entity Recognition (NER) with Image Classification to validate textual descriptions of images. Given a user input like:

📝 Text: "There is a cat in the picture."
🖼 Image: (an actual image containing an animal)

The system will determine whether the described animal matches the one in the image.

🔹 Key Features

✅ Named Entity Recognition (NER) – Extracts animal names from textual descriptions.
✅ Image Classification – Identifies animals in images using a pre-trained deep learning model.
✅ Multi-Modal Analysis – Combines NLP and Computer Vision for better validation.
✅ Pre-Trained Models – Uses fine-tuned models for both NER and image classification.
✅ Custom Dataset Generation – NER dataset is generated dynamically, while image dataset is sourced from Kaggle.

📂 Project Structure
NER_Image_Classification/
│
├── dataset/  # Data storage (NER dataset included, image dataset needs to be downloaded)
│   ├── ner_data.json  # Generated dataset for NER
│
├── NER/  # Named Entity Recognition Model
│   ├── model/  # Contains trained model files (download required)
│   ├── trainer_ner.py  # Script for training the NER model
│   ├── inference_ner.py  # Inference script for NER
│
├── Image_Classification/  # Image classification model
│   ├── model/  # Contains trained model files (download required)
│   ├── trainer_image.py  # Script for training image classifier
│   ├── inference_image.py  # Inference script for classification
│
├── Pipeline/  # Integrated pipeline for inference
│   ├── pipeline.py  # Main script that combines NER and image classification
│
├── notebooks/  # Jupyter Notebooks for data exploration and demos
│   ├── eda.ipynb  # Exploratory Data Analysis
│   ├── draft.ipynb  # Draft Notebook
│   ├── demo.ipynb  # Demonstration notebook
│
├── scripts/  # Utility scripts
│   ├── generate_ner_data.py  # Script to generate synthetic NER dataset
│
├── requirements.txt  # Dependencies
├── README.md  # Documentation

🔧 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Markol06/AI-DS-testproject.git
cd AI-DS-testproject/NER_Image_Classification
2️⃣ Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
3️⃣ Download trained models from Hugging Face
from huggingface_hub import snapshot_download

# Download NER Model
snapshot_download(repo_id="Markol06/ner_model", local_dir="NER/model")

# Download Image Classification Model
snapshot_download(repo_id="Markol06/image_classification_model", local_dir="Image_Classification/model")

📥 Downloading the Dataset

The NER dataset (ner_data.json) is already included in the repository. However, you need to download the Animals-10 dataset for image classification.
🔹 How to Download and Set Up animals10

1. Download the dataset from Kaggle:
https://www.kaggle.com/datasets/alessiocorrado99/animals10
2. Extract the dataset into the dataset/animals10/ folder
mkdir -p dataset/animals10
unzip animals10.zip -d dataset/animals10/
3.Ensure the dataset structure is as follows (and folders with animals are translated to english):
dataset/
├── animals10/
│   ├──raw-img
│   │  ├── butterfly/
│   │  ├── cat/
│   │  ├── chicken/
│   │  ├── cow/
│   │  ├── dog/
│   │  ├── elephant/
│   │  ├── horse/
│   │  ├── sheep/
│   │  ├── spider/
│   │  ├── squirrel/
├── ner_data.json  # Dataset for NER

4️⃣ Run the Demo
🏗 Option 1: Run in Jupyter Notebook

jupyter notebook

Then open notebooks/demo.ipynb and run all cells.

🏗 Option 2: Run Directly in Python

python Pipeline/pipeline.py

This script will take user input and check if the described animal matches the one in the image.

🎯 How the Solution Works

1️⃣ Named Entity Recognition (NER)

Extracts animal names from text.

Uses a transformer-based model fine-tuned on a custom dataset.

Training Script: trainer_ner.py trains the model on labeled text data.

Inference Script: inference_ner.py loads the trained model to extract entities.

2️⃣ Image Classification

Identifies animals in images using a deep learning model (ResNet-50).

Trained on animals10 dataset with 10 classes.

Training Script: trainer_image.py trains the classification model.

Inference Script: inference_image.py predicts the animal class in an image.

3️⃣ Full Pipeline Execution

Combines both models to validate user input against an image.
I added testimage.png so you can test the pipeline.py script.
So, everything works like in the described flow:
In general, the flow should be the following:
1. The user provides a text similar to “There is a cow in the picture.” and an image that - **you have to put input without "" for both sentence and path to image**
contains any animal.
2. Your pipeline should decide if it is true or not and provide a boolean value as the output.
You should take care that the text input will not be the same as in the example, and the
user can ask it in a different way.

📊 Model Performance
NER Model: F1-score of 0.99, indicating highly accurate entity recognition.

Image Classification Model: F1-score of 0.81, performing well but with room for improvement.
