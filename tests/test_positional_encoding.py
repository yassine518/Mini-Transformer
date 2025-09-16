import torch
from src.positional_encoding import PositionalEncoding


def test_positional_encoding_shapes():
    batch, seq_len, d_model = 2, 10, 16
    pe = PositionalEncoding(d_model=d_model, max_len=50)

    x = torch.zeros(batch, seq_len, d_model)
    out = pe(x)

    assert out.shape == (batch, seq_len, d_model)
    print("PositionalEncoding test passed ✅")


if __name__ == "__main__":
    test_positional_encoding_shapes()
