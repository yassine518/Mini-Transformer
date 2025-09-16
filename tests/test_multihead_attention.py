import torch
from src.multihead_attention import MultiHeadAttention


def test_multihead_attention_shapes():
    batch, seq_len, d_model, n_heads = 2, 5, 16, 4
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    x = torch.rand(batch, seq_len, d_model)
    out, attn = mha(x, x, x)

    assert out.shape == (batch, seq_len, d_model)
    assert attn.shape == (batch, n_heads, seq_len, seq_len)


if __name__ == "__main__":
    test_multihead_attention_shapes()
    print("MultiHeadAttention test passed ✅")
