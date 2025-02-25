import random
import json
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_dir = os.path.join(project_root, "dataset")
dataset_path = os.path.join(dataset_dir, "ner_data.json")

os.makedirs(dataset_dir, exist_ok=True)

def correct_article(animal):
    if animal[0] in "aeiou":  # якщо перша літера голосна
        return "an " + animal
    return "a " + animal

# 100 unique animal classes
animals = [
    "cat", "dog", "cow", "tiger", "elephant", "giraffe", "horse", "lion", "bear", "zebra",
    "wolf", "fox", "rabbit", "deer", "kangaroo", "panda", "monkey", "goat", "sheep",
    "crocodile", "dolphin", "shark", "penguin", "squirrel", "butterfly", "spider", "chicken",
    "octopus", "jaguar", "hippopotamus", "flamingo", "beaver", "peacock", "vulture", "raccoon", "lynx", "hyena",
    "cobra", "bison", "parrot", "owl", "antelope", "meerkat", "armadillo", "stingray", "cockatoo", "boar",
    "hedgehog", "ostrich", "platypus", "walrus", "mantis", "starfish", "tapir", "gazelle", "mole", "sloth",
    "newt", "salmon", "porcupine", "chameleon", "barracuda", "weasel", "quokka", "donkey", "moose", "puffin",
    "whale", "crab", "eel", "ferret", "puma", "seal", "shrimp", "turkey", "yak", "gecko",
    "lobster", "mongoose", "scorpion", "snail", "tortoise", "viper", "wombat", "albatross", "clownfish", "dingo",
    "emu", "falcon", "gopher", "iguana", "krill", "marmot", "narwhal", "oyster", "stingray", "tarsier"
]

# 100 unique sentence examples
sentence_templates = [
    "I see a {0} in the picture.", "There is a {0} in the image.", "Could you tell me if this is a {0}?",
    "Looks like a {0}, doesn't it?", "I think this is a {0}.", "Do you see a {0} here?",
    "This photo has a {0}, right?", "I believe this is a {0} in the frame.", "Check if this is a {0}.",
    "A {0} is clearly visible in this picture.", "A {0} appears to be in the background.",
    "I spotted a {0} in this photo.", "What do you think? Is this a {0}?", "The image contains a {0}, correct?",
    "A {0} seems to be present in this shot.", "Is that a {0} in the background?",
    "This looks like a {0}, what do you think?", "I have a feeling that this is a {0}.",
    "There seems to be a {0} over there.", "It looks like a {0} to me.", "This must be a {0}, right?",
    "A {0} is standing right there.", "It is hard to tell, but this could be a {0}.",
    "Maybe there is a {0} hiding in this image?", "This photo could feature a {0}, don't you think?",
    "Could you double-check if there is a {0}?", "It seems to resemble a {0}.",
    "This might be a {0}, or am I wrong?", "Isn't that a {0} in the frame?",
    "Something tells me this is a {0}.", "Would you say this is a {0}?",
    "This could be a perfect shot of a {0}.", "Are we looking at a {0} here?",
    "What animal is this? Maybe a {0}?", "If I am not mistaken, this is a {0}.",
    "This certainly appears to be a {0}.", "The shape of this suggests it is a {0}.",
    "A {0} can be seen in the distance.", "Could this image contain a {0}?",
    "This might just be a {0}.", "I'm not sure, but it looks like a {0}.",
    "Does this picture show a {0}?", "A {0} is definitely present in this picture.",
    "You can notice a {0} in the corner.", "Isn't there a {0} somewhere here?",
    "Tell me if this is a {0}.", "I bet this is a {0}.", "A {0} is somewhere in this shot.",
    "This image must have a {0}.", "A {0} is in this environment.",
    "I suspect there is a {0} here.", "Could this really be a {0}?",
    "Can you point out the {0}?", "A {0} is almost hidden in this photo.",
    "It looks a lot like a {0}.", "A {0} can barely be seen in the frame.",
    "You can faintly see a {0} in the distance.", "Am I wrong or is this a {0}?",
    "A {0} is sitting right in the middle of the image.",
    "This photo reminds me of a {0}.", "A {0} is quietly present in this image.",
    "Tell me whether you think this is a {0}.", "In the middle, a {0} appears.",
    "Is it obvious that this is a {0}?", "There’s a {0} right there.",
    "I can’t tell if this is a {0} or not.", "That definitely looks like a {0}.",
    "Do you recognize this as a {0}?", "It looks as though there’s a {0} here.",
    "This frame includes a {0}.", "I wasn’t sure, but I think it’s a {0}.",
    "There must be a {0} hiding there.", "Can you distinguish the {0}?",
    "A {0} stands out in this shot.", "Take a close look, is that a {0}?",
    "This resembles a {0}, don’t you think?", "A {0} is probably in this image.",
    "Do you think there is a {0} here?",
    "Clearly, we have a {0} in this image.",
    "I'm guessing this could be a {0}.",
    "Can you confirm if this is a {0}?",
    "This surely looks like a {0}, right?",
    "Is a {0} hiding somewhere in the background?",
    "The picture likely features a {0}.",
    "There's definitely a {0} in sight.",
    "Wouldn't you agree this is a {0}?",
    "Perhaps there's a {0} captured here?"
]

def generate_ner_data(num_sentences=10000):
    dataset = []
    for _ in range(num_sentences):
        animal = random.choice(animals)
        sentence = random.choice(sentence_templates).format(animal)

        # Determining the position of a word in the text
        entity = {
            "text": animal,
            "start": sentence.index(animal),
            "end": sentence.index(animal) + len(animal),
            "label": "ANIMAL"
        }

        dataset.append({"sentence": sentence, "entities": [entity]})

    return dataset

# Generating 10 000 sentences
ner_dataset = generate_ner_data(10000)

# Saving in JSON
with open(dataset_path, "w") as f:
    json.dump(ner_dataset, f, indent=4)

print(f"✅ Saved {len(ner_dataset)} sentences in {dataset_path}")

