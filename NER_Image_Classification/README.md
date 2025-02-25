Named Entity Recognition + Image Classification Pipeline


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
