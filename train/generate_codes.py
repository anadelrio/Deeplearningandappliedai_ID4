# train/generate_codes.py
import os
import numpy as np
import torch
import torch.nn as nn

# Config
MODEL_PATH = "models/autoregressive_model.pt"
OUT_PATH = "experiments/latent-spaces/generated_codes.npy"
SEQ_LEN = 32
N_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo autoregresivo (debe coincidir con train_autoregressive.py)
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

# Instanciar y cargar modelo
model = CodeRNN(vocab_size=N_CLASSES, embed_dim=64, hidden_dim=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Secuencia inicial (semilla)
generated = [np.random.randint(0, N_CLASSES)]  # seed aleatorio

# Generar códigos uno a uno
for _ in range(SEQ_LEN - 1):
    input_seq = torch.tensor(generated[-32:], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_seq)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
    generated.append(next_token)

# Guardar códigos generados
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.save(OUT_PATH, np.array(generated))
print(f"Generated codes saved to {OUT_PATH}")
