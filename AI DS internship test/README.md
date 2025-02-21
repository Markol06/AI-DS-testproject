Image Classification + OOP (MNIST)
File structure:
📌 Project Overview – a general description of what the project is and what it is for.
🛠 Technologies Used – list of technologies and libraries.
📁 Project Structure – explanation of folder structure.
🚀 Installation & Setup – a detailed guide on how to launch a project.
🎯 How the Solution Works – explanation of how the code works.
🔍 Testing & Edge Cases – edge case testing results.
📊 Results & Analysis – conclusions about the accuracy of the models.

📌 Project Overview

This project implements three classification models for the MNIST dataset:
1️⃣ Random Forest (rf) – a non-neural machine learning model.
2️⃣ Feed-Forward Neural Network (NN) (nn) – a basic deep learning model.
3️⃣ Convolutional Neural Network (CNN) (cnn) – the most advanced model.

Each model follows Object-Oriented Programming (OOP) principles and implements a common interface (MnistClassifierInterface).
A wrapper class (MnistClassifier) allows seamless switching between models.

The project includes:
✔ A structured Python implementation of MNIST classification models.
✔ Object-oriented design to maintain modularity.
✔ Jupyter Notebook (demo.ipynb) showcasing model performance and edge cases.
✔ Edge case testing for robustness evaluation.

🛠 Technologies Used

Python 3.10

TensorFlow 2.13.0 (Neural Networks & CNNs)

Scikit-learn 1.3.0 (Random Forest Classifier)

NumPy 1.24.3 (Data manipulation)

Matplotlib 3.7.1 (Visualizations)

Seaborn 0.12.2 (Advanced statistical plots)

OpenCV 4.8.0.76 (Edge case image modifications)

📁 Project Structure

📂 Image_classification_OOP
 ├── 📂 Winstars AI DS internship test
 │   ├── 📂 models
 │   │   ├── mnist_classifier_interface.py  # Interface for models
 │   │   ├── random_forest_classifier.py    # Random Forest implementation
 │   │   ├── neural_network_classifier.py   # Neural Network implementation
 │   │   ├── cnn_classifier.py              # CNN implementation
 │   │   ├── mnist_classifier.py            # Wrapper for all models
 │   │   ├── __init__.py
 │   ├── 📂 notebooks
 │   │   ├── demo.ipynb  # Jupyter Notebook with full testing & visualization
 │   │   ├── draft.ipynb  # Draft Notebook where I tested the code.
 │   ├── requirements.txt  # Required libraries
 │   ├── README.md  # Project documentation

🚀 Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/Markol06/AI-DS-testproject.git
cd AI-DS-testproject/Image_classification_OOP/Winstars AI DS internship test

2️⃣ Install dependencies

Run the following command to install all required libraries:

pip install -r requirements.txt

3️⃣ Run the demo

Option 1: Run in Jupyter Notebook

jupyter notebook

Then open notebooks/demo.ipynb and run all cells.

Option 2: Run directly in Python

python models/mnist_classifier.py

The script will train and test models automatically.

🎯 How the Solution Works

1️⃣ Model Architecture

Each model follows an interface (MnistClassifierInterface) that defines:

train(X_train, y_train): Method to train the model.

predict(X_test): Method to make predictions.

Models implemented:

Random Forest (RF) – traditional ML classifier, works with flattened images.

Neural Network (NN) – basic deep learning model with dense layers.

CNN – convolutional layers extract spatial features for improved accuracy.

2️⃣ Model Selection via MnistClassifier

To use a specific model, simply initialize:

classifier = MnistClassifier("cnn")  # Options: "cnn", "nn", "rf"
classifier.train()
prediction = classifier.predict(X_test[0])

🔍 Testing & Edge Cases

The project tests models on challenging cases:

Blurred digits: CNN and NN performed best, while RF failed.

Cropped digits: All models recognized cropped digits correctly.

Resized digits: CNN handled scaling well, while RF and NN struggled.

Skewed digits: All models had difficulty with skewed inputs.

Noisy digits: All models classified noisy digits correctly.

Darkened digits: Even minor darkening caused misclassification in all models.

Brightened digits: Overexposed digits led to incorrect results.

📊 Results & Analysis

Random Forest achieved ~97.0% accuracy in 32 seconds.

Neural Network achieved ~97.8% accuracy in 20 seconds.

CNN achieved ~99.2% accuracy, but took 515 seconds to train.

Key Findings:
CNN provides the best accuracy but has the longest training time. Random Forest is the fastest but less accurate. All models struggle with darkened and brightened digits.

👨‍💻 Author & Contributions

Project created by Marko as part of the AI-DS internship test.


