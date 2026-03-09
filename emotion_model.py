import os
import librosa
import numpy as np


DATA_PATH = "data/ravdess/audio_speech_actors_01-24"

actors = os.listdir(DATA_PATH)
print("Actor folders found:", len(actors))

first_actor = os.path.join(DATA_PATH, actors[0])
files = os.listdir(first_actor)

print("Sample audio files:", files[:5])

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

sample_file = os.path.join(first_actor, files[0])
features = extract_mfcc(sample_file)

print("MFCC feature shape:", features.shape)
print("First 5 MFCC values:", features[:5])

emotion_map = {
    "01": 0,  # neutral
    "03": 1,  # happy
    "04": 2,  # sad
    "05": 3   # angry
}

X = []
y = []

for actor in actors:
    actor_path = os.path.join(DATA_PATH, actor)
    for file in os.listdir(actor_path):
        emotion_code = file.split("-")[2]
        if emotion_code in emotion_map:
            file_path = os.path.join(actor_path, file)
            features = extract_mfcc(file_path)
            X.append(features)
            y.append(emotion_map[emotion_code])

print("Total samples collected:", len(X))

import torch

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


import torch.nn as nn
import torch.optim as optim

class EmotionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = EmotionNN()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():
    predictions = torch.argmax(model(X_test), dim=1)
    accuracy = (predictions == y_test).float().mean()

print("Base Model Accuracy:", accuracy.item())




class Attention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.att = nn.Linear(features, features)

    def forward(self, x):
        weights = torch.softmax(self.att(x), dim=1)
        return x * weights

class EmotionAttentionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention(40)
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.attention(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

att_model = EmotionAttentionNN()
optimizer = optim.Adam(att_model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()

    outputs = att_model(X_train)
    loss = loss_fn(outputs, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"[Attention] Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():
    predictions = torch.argmax(att_model(X_test), dim=1)
    accuracy = (predictions == y_test).float().mean()

print("Attention Model Accuracy:", accuracy.item())
