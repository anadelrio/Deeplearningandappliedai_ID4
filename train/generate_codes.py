import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Configuration
LABELS_PATH = "experiments/latent-spaces/geodesic_labels.npy"
MODEL_PATH = "models/autoregressive_model.pt"
OUTPUT_PATH = "experiments/latent-spaces/generated_codes.npy"

SEQ_LEN = 16         # Context length used during training
GENERATE_LEN = 32    # Number of new tokens to generate
NUM_CLASSES = 10     # Number of discrete codes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Autoregressive model (same as used for training)
class CodeRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))

# Load the trained model
model = CodeRNN(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load initial seed from real data
labels = np.load(LABELS_PATH)
seed = labels[:SEQ_LEN].tolist()  # Use the first tokens as starting context

generated = seed.copy()

print("Generating new token sequence...")
for _ in tqdm(range(GENERATE_LEN)):
    context = torch.tensor([generated[-SEQ_LEN:]], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(context)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)

# Save generated tokens (excluding seed)
generated = np.array(generated[SEQ_LEN:])
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.save(OUTPUT_PATH, generated)
print(f"Generated codes saved to: {OUTPUT_PATH}")

