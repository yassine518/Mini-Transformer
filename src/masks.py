import torch

def create_padding_mask(seq, pad_idx=0):
    """
    Mask for padding tokens.
    seq: (batch_size, seq_len)
    Returns mask of shape (batch_size, 1, 1, seq_len)
    """
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # True = masked

def create_look_ahead_mask(seq_len):
    """
    Mask future tokens in decoder.
    Returns (1, 1, seq_len, seq_len)
    """
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(1)
