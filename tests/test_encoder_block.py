import torch
from src.encoder_block import EncoderBlock


def test_encoder_block_shapes():
    batch, seq_len, d_model, n_heads, d_ff = 2, 10, 16, 4, 64
    encoder_block = EncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

    x = torch.rand(batch, seq_len, d_model)
    out = encoder_block(x)

    assert out.shape == (batch, seq_len, d_model)
    print("EncoderBlock test passed ✅")


if __name__ == "__main__":
    test_encoder_block_shapes()
