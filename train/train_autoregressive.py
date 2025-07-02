import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Config
LABELS_PATH = "experiments/latent-spaces/geodesic_labels.npy"
MODEL_OUT = "models/autoregressive_model.pt"
SEQ_LEN = 16              # longitud de contexto
BATCH_SIZE = 64
EPOCHS = 20
EMBED_DIM = 32
HIDDEN_DIM = 128
N_CLASSES = 10            # K-means clusters = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class CodeSequenceDataset(Dataset):
    def __init__(self, labels, seq_len):
        self.inputs = []
        self.targets = []
        for i in range(len(labels) - seq_len):
            self.inputs.append(labels[i:i+seq_len])
            self.targets.append(labels[i+seq_len])
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Autoregressive model: Embedding + GRU + FC
class CodeRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)                       # (B, T, D)
        _, h = self.rnn(x)                      # h: (1, B, H)
        out = self.fc(h.squeeze(0))             # (B, V)
        return out

# Chargue discrete labels
labels = np.load(LABELS_PATH).astype(np.int64)
dataset = CodeSequenceDataset(labels, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Inicialize model
model = CodeRNN(N_CLASSES, EMBED_DIM, HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# train
print("Training autoregressive model...")
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        logits = model(x)                       # (B, V)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

# save model
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
torch.save(model.state_dict(), MODEL_OUT)
print(f"Model saved at: {MODEL_OUT}")
