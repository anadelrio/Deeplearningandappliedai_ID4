import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Config
CODES_PATH = "experiments/latent-spaces/geodesic_codes.npy"
MODEL_OUT = "models/autoregressive_model.pt"
SEQ_LEN = 32  # How many previous tokens to use
BATCH_SIZE = 64
EPOCHS = 10
EMBED_DIM = 64
HIDDEN_DIM = 128
N_CLASSES = 10  # Number of unique codes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class CodeSequenceDataset(Dataset):
    def __init__(self, codes, seq_len):
        self.inputs = []
        self.targets = []
        for i in range(len(codes) - seq_len):
            self.inputs.append(codes[i:i+seq_len])
            self.targets.append(codes[i+seq_len])
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Model
class CodeRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.rnn(x)
        out = self.fc(h.squeeze(0))
        return out

# Load data
codes = np.load(CODES_PATH)
dataset = CodeSequenceDataset(codes, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = CodeRNN(N_CLASSES, EMBED_DIM, HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
print("Training autoregressive model...")
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")
