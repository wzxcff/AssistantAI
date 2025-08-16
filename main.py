import random

import torch.nn as nn
from nltk import WordNetLemmatizer
import nltk
import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from logic import *


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class Assistant:
    def __init__(self, intents_path, functions_mapping=None):
        self.vocab = []
        self.intents = []
        self.intents_responses = {}
        self.documents = []
        self.intents_path = intents_path

        self.X = None
        self.y = None

        self.function_mappings = functions_mapping

    def tokenize_and_lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    def parse_intents(self):
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError("Intents can't be found")

        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)

        for intent in intents_data["intents"]:
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])
                self.intents_responses[intent["tag"]] = intent["responses"]

            for pattern in intent["patterns"]:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocab.extend(pattern_words)
                self.documents.append((pattern_words, intent["tag"]))

        self.vocab = sorted(set(self.vocab))

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocab]

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            bags.append(bag)

            intent_index = self.intents.index(document[1])
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = Model(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss
                print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({"input_size": self.X.shape[1], "output_size": len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):

        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = Model(dimensions["input_size"], dimensions["output_size"])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor(bag, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

            predicted_class_index = torch.argmax(predictions, dim=1).item()
            predicted_intent = self.intents[predicted_class_index]

            if self.function_mappings:
                if predicted_intent in self.function_mappings:
                    return self.function_mappings[predicted_intent]()

            if self.intents_responses[predicted_intent]:
                return random.choice(self.intents_responses[predicted_intent])
            else:
                return predicted_intent


if __name__ == '__main__':

    training_mode = False
    function_mapping = {"weather": weather, "reminder": reminder, "math": math}

    if training_mode:
        print("Training mode activated")
        assistant = Assistant("intents.json", function_mapping)
        assistant.parse_intents()
        assistant.prepare_data()
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)
        assistant.save_model(model_path="assistant_model.pth", dimensions_path="dimensions.json")
    else:
        print("Loading model")
        assistant = Assistant("intents.json", function_mapping)
        assistant.parse_intents()
        assistant.load_model("assistant_model.pth", "dimensions.json")

    while True:
        message = input("> ")

        if message == "q":
            break

        print(assistant.process_message(message))
