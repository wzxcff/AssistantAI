import torch.nn as nn
from nltk import WordNetLemmatizer
import nltk
import os
import json


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_of_classes):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_of_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Assistant:
    def __init__(self, intents_path):
        self.vocab = []
        self.intents = []
        self.intents_responses = {}
        self.documents = []

    def tokenize_and_lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words
