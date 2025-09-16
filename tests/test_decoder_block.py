import torch
from src.decoder_block import DecoderBlock


def test_decoder_block_shapes():
    batch, seq_len, d_model, n_heads, d_ff = 2, 10, 16, 4, 64
    decoder_block = DecoderBlock(d_model, n_heads, d_ff)

    x = torch.rand(batch, seq_len, d_model)           # target input
    enc_output = torch.rand(batch, seq_len, d_model)  # encoder output

    out = decoder_block(x, enc_output)

    assert out.shape == (batch, seq_len, d_model)
    print("DecoderBlock test passed ✅")


if __name__ == "__main__":
    test_decoder_block_shapes()
