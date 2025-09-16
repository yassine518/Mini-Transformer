import torch
import torch.nn as nn
import torch.optim as optim
from src.transformer import Transformer
from src.masks import create_padding_mask, create_look_ahead_mask

# -------------------------
# Hyperparameters
# -------------------------
SRC_VOCAB_SIZE = 20
TGT_VOCAB_SIZE = 20
SEQ_LEN = 5
BATCH_SIZE = 32
D_MODEL = 32
N_HEADS = 4
D_FF = 64
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
EPOCHS = 200
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_IDX = 0

# -------------------------
# Toy dataset (copy task)
# -------------------------
def generate_batch(batch_size, seq_len, vocab_size, pad_idx=PAD_IDX):
    # Random sequences from 1 to vocab_size-1 (0 reserved for padding)
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt = src.clone()
    return src, tgt

# -------------------------
# Create model
# -------------------------
model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, N_HEADS, D_FF,
                    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training loop
# -------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    src, tgt = generate_batch(BATCH_SIZE, SEQ_LEN, SRC_VOCAB_SIZE)
    src, tgt = src.to(DEVICE), tgt.to(DEVICE)

    optimizer.zero_grad()

    # Create masks
    src_mask = create_padding_mask(src, pad_idx=PAD_IDX)
    tgt_mask = create_padding_mask(tgt[:, :-1], pad_idx=PAD_IDX) | create_look_ahead_mask(tgt[:, :-1].size(1))

    # Forward pass
    output = model(src, tgt[:, :-1], src_mask=src_mask, tgt_mask=tgt_mask)

    # Reshape for CrossEntropyLoss
    output = output.reshape(-1, TGT_VOCAB_SIZE)
    tgt_labels = tgt[:, 1:].reshape(-1)

    # Compute loss and backprop
    loss = criterion(output, tgt_labels)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.4f}")

print("Training finished ✅")
