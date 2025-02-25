import json
import os
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from nltk.corpus import wordnet
from itertools import chain
from sklearn.metrics import precision_score, recall_score, f1_score

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = os.path.join(project_root, "dataset", "ner_data.json")
model_dir = os.path.join(project_root, "NER", "model")

# Model configuration
model_name = "distilbert-base-cased"
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=2,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

# Data loading and preprocessing
def load_and_prepare_data():
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["sentence"] for item in data]
    entities = [[(ent["start"], ent["end"], ent["label"]) for ent in item["entities"]] for item in data]

    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_offsets_mapping=True)

    labels = []
    for i, offset_mapping in enumerate(tokenized_inputs.offset_mapping):
        label = np.zeros(len(offset_mapping), dtype=int)
        for start, end, _ in entities[i]:
            for idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start >= start and token_end <= end:
                    label[idx] = 1
        labels.append(label)

    tokenized_inputs["labels"] = labels
    return texts, labels

# Data augmentation methods
def shuffle_words(sentence):
    words = sentence.split()
    random.shuffle(words)
    return ' '.join(words)

def align_labels(labels, encodings):
    aligned_labels = []
    for label, input_ids in zip(labels, encodings['input_ids']):
        padded_label = np.zeros(len(input_ids), dtype=int)
        length = min(len(label), len(input_ids))
        padded_label[:length] = label[:length]
        aligned_labels.append(padded_label.tolist())
    return aligned_labels

# Dataset class for NER
def create_dataset(texts, labels):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=50)
    aligned_labels = align_labels(labels, encodings)

    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    return NERDataset(encodings, aligned_labels)

# Training function
def train_ner_model():
    texts, labels = load_and_prepare_data()

    # Data augmentation
    augmented_texts = []
    augmented_labels = []
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)

        shuffled_text = shuffle_words(text)
        augmented_texts.append(shuffled_text)
        augmented_labels.append(label)

    # Using only 30% of the dataset
    _, texts_30, _, labels_30 = train_test_split(
        augmented_texts, augmented_labels, test_size=0.3, random_state=42
    )

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts_30, labels_30, test_size=0.2, random_state=42
    )

    train_dataset = create_dataset(train_texts, train_labels)
    test_dataset = create_dataset(test_texts, test_labels)

    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'logs'), exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir=os.path.join(model_dir, 'logs'),
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=2,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"✅ Model and tokenizer saved to {model_dir}")

    # Evaluate the model
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    # Flatten labels and predictions
    true_labels_flat = np.concatenate([item['labels'].numpy().flatten() for item in test_dataset])
    pred_labels_flat = preds.flatten()

    # Align label lengths
    min_len = min(len(true_labels_flat), len(pred_labels_flat))
    true_labels_flat = true_labels_flat[:min_len]
    pred_labels_flat = pred_labels_flat[:min_len]

    # Compute metrics
    precision = precision_score(true_labels_flat, pred_labels_flat, average="binary")
    recall = recall_score(true_labels_flat, pred_labels_flat, average="binary")
    f1 = f1_score(true_labels_flat, pred_labels_flat, average="binary")

    print(f"📊 Evaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    train_ner_model()