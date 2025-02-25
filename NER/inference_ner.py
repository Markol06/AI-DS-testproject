import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Define project structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(project_root, "NER", "model")

# Load tokenizer and model for inference
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

# Create inference pipeline
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)


# Function to extract animal name from text
def extract_animal_name(sentence):
    results = ner_pipeline(sentence)

    # Collecting words labeled as "LABEL_1"
    animal_tokens = []
    for res in results:
        if res['entity'] == "LABEL_1":
            word = res['word'].replace("##", "")  # Fix subwords
            animal_tokens.append(word)

    if not animal_tokens:
        return None  # No animal detected

    # Join tokens to form a complete word
    animal_name = "".join(animal_tokens)
    return animal_name


# Example usage
if __name__ == "__main__":
    example_sentences = [
        "There is a cow in this picture.",
        "I think there might be a horse here.",
        "Do you see a penguin in the image?"
    ]

    for sentence in example_sentences:
        detected_animal = extract_animal_name(sentence)
        print(f"Sentence: {sentence}")
        print(f"Detected animal: {detected_animal}")