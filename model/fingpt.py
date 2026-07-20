"""
FinGPT-Tiny — GPT-style Language Model from Scratch
Decoder-only transformer for financial text generation
Architecture: RoPE embeddings, RMSNorm, SwiGLU activation
Same technical depth as the medical SLM post
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — more stable than LayerNorm"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) — used in Llama, GPT-NeoX"""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class SwiGLU(nn.Module):
    """SwiGLU activation — used in PaLM, Llama"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """
    Causal multi-head self attention with RoPE
    Causal = each token can only attend to previous tokens
    This is what makes it a language MODEL (generator) not a classifier
    """
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 512):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.dim = dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.d_k, max_seq_len)

        # Causal mask — lower triangular
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(x, T)
        Q, K = apply_rotary(Q, K, cos, sin)

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single GPT block: RMSNorm + Attention + RMSNorm + SwiGLU"""
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, max_seq_len: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FinGPT(nn.Module):
    """
    FinGPT-Tiny: GPT-style language model for financial text
    Trained from scratch — no pretrained weights
    Architecture inspired by Llama: RoPE, RMSNorm, SwiGLU

    Input:  token ids (batch, seq_len)
    Output: logits over vocabulary (batch, seq_len, vocab_size)
    """
    def __init__(
        self,
        vocab_size: int = 16000,
        dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        hidden_dim = int(dim * 2.67)  # SwiGLU hidden expansion

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim, max_seq_len)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying — token embedding and LM head share weights
        # Same technique used in GPT-2, Llama
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape

        x = self.token_emb(idx)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=0  # ignore padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        """Generate financial text autoregressively"""
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx[:, -512:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = FinGPT(
        vocab_size=16000,
        dim=256,
        num_heads=8,
        num_layers=6,
        max_seq_len=512,
    )

    total_params = model.count_parameters()
    print(f"FinGPT-Tiny Parameters: {total_params:,}")
    print(f"Approx size: {total_params * 4 / 1e6:.1f} MB (float32)")

    # Forward pass test
    x = torch.randint(0, 16000, (2, 128))
    targets = torch.randint(0, 16000, (2, 128))
    logits, loss = model(x, targets)

    print(f"\nInput shape:  {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss:         {loss.item():.4f}")
    print(f"Expected loss (random): {torch.log(torch.tensor(16000.0)):.4f}")

    # Generation test
    prompt = torch.randint(0, 16000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"\nGeneration test:")
    print(f"Prompt length: {prompt.shape[1]}")
    print(f"Generated length: {generated.shape[1]}")
    print("Forward pass and generation successful!")