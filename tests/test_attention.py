import torch
from src.attention import ScaledDotProductAttention


def test_attention_shapes():
    batch, seq_len, d_k = 2, 4, 8
    q = torch.rand(batch, seq_len, d_k)
    k = torch.rand(batch, seq_len, d_k)
    v = torch.rand(batch, seq_len, d_k)

    attn = ScaledDotProductAttention()
    output, weights = attn(q, k, v)

    assert output.shape == (batch, seq_len, d_k)
    assert weights.shape == (batch, seq_len, seq_len)


if __name__ == "__main__":
    test_attention_shapes()
    print("ScaledDotProductAttention test passed ✅")
