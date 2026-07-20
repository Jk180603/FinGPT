"""
FinGPT-Tiny — Streamlit Dashboard
Interactive UI for text generation, evaluation results, and model exploration
"""
import streamlit as st
import torch
import json
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
sys.path.insert(0, ".")

st.set_page_config(
    page_title="FinGPT-Tiny",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .stApp { background-color: #0f1117; }
  .metric-card {
    background: #1a1d29; border: 1px solid #2d3142;
    border-radius: 8px; padding: 16px; text-align: center;
  }
  code { color: #4ade80; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    from tokenizers import Tokenizer
    from model.fingpt import FinGPT
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("tokenizer/financial_bpe.json")
    model = FinGPT(
        vocab_size=16000, dim=256, num_heads=8, num_layers=6, max_seq_len=256
    ).to(device)
    model.load_state_dict(torch.load(
        "model/checkpoints/best_model.pt", map_location=device))
    model.eval()
    return model, tokenizer, device


# ── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.title("FinGPT-Tiny")
st.sidebar.caption("GPT-style language model trained from scratch on 53.8MB of financial text")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["Generate", "Evaluation", "Model Info"])
st.sidebar.divider()
st.sidebar.markdown("""
**Architecture**
- 8.8M parameters
- 6 transformer layers
- 8 attention heads
- 256 hidden dimensions
- RoPE + RMSNorm + SwiGLU
- Weight tying

**Training**
- 53.8MB financial corpus
- 16K BPE vocabulary
- 5 epochs
- Best Val PPL: 50.2
""")

# ── PAGES ─────────────────────────────────────────────────────
if page == "Generate":
    st.title("📈 FinGPT-Tiny — Financial Text Generation")
    st.caption("A GPT-style language model trained from scratch on financial text")

    try:
        model, tokenizer, device = load_model()

        col1, col2 = st.columns([2, 1])
        with col1:
            prompt = st.text_area(
                "Enter a financial prompt",
                value="Revenue grew 23% year over year to",
                height=80
            )
        with col2:
            temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05,
                help="Higher = more creative, lower = more conservative")
            max_tokens = st.slider("Max new tokens", 20, 200, 80, 10)
            top_k = st.slider("Top-k", 10, 100, 50, 5)

        if st.button("Generate", type="primary", use_container_width=True):
            with st.spinner("Generating..."):
                ids = tokenizer.encode(prompt).ids
                x = torch.tensor([ids], dtype=torch.long).to(device)
                with torch.no_grad():
                    out = model.generate(x, max_new_tokens=max_tokens,
                                        temperature=temperature, top_k=top_k)
                text = tokenizer.decode(out[0].tolist())

            st.divider()
            st.subheader("Output")
            prompt_end = len(prompt)
            st.markdown(f"""
            <div style="background:#1a1d29;border:1px solid #2d3142;border-radius:8px;padding:16px;font-family:monospace;line-height:1.8">
              <span style="color:#c9d4e0">{prompt}</span><span style="color:#4ade80">{text[prompt_end:]}</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Generated {len(out[0]) - len(ids)} new tokens")

        st.divider()
        st.subheader("Quick examples")
        examples = [
            "Revenue grew 23% year over year to",
            "The company reported strong EBITDA margins of",
            "Free cash flow was $1.8 billion, up from",
            "We are raising full-year EPS guidance to",
            "The Board of Directors approved a $5 billion",
            "Net income attributable to common shareholders",
        ]
        cols = st.columns(3)
        for i, ex in enumerate(examples):
            with cols[i % 3]:
                if st.button(ex[:35] + "...", key=f"ex{i}"):
                    ids = tokenizer.encode(ex).ids
                    x = torch.tensor([ids], dtype=torch.long).to(device)
                    with torch.no_grad():
                        out = model.generate(x, max_new_tokens=60,
                                            temperature=0.8, top_k=50)
                    text = tokenizer.decode(out[0].tolist())
                    st.markdown(f"**{ex}**")
                    st.write(text[len(ex):])

    except Exception as e:
        st.error(f"Model not loaded: {e}")
        st.info("Run `python model/pretrain.py` first to train the model.")

elif page == "Evaluation":
    st.title("📊 Evaluation Results")

    if os.path.exists("model/eval_results.json"):
        with open("model/eval_results.json") as f:
            results = json.load(f)
        ev = results["evaluation"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Perplexity", f"{ev['test_perplexity']:.1f}",
                   help="Lower is better. Random baseline = 16,000")
        col2.metric("Financial Knowledge", f"{ev['financial_knowledge_accuracy']:.1%}",
                   help="Fill-in-the-blank accuracy on financial terms")
        col3.metric("Vocab Diversity", f"{ev['vocab_diversity']:.3f}",
                   help="Unique token ratio (0-1, higher = more diverse)")
        col4.metric("Repetition Rate", f"{ev['repetition_rate']:.3f}",
                   help="Bigram repetition (lower = less repetitive)")

        st.divider()

        # Perplexity comparison chart
        st.subheader("Perplexity vs Baselines")
        baselines = {
            "Random (16K vocab)": 16000,
            "Untrained model": 9.68,
            "FinGPT-Tiny (ours)": ev["test_perplexity"],
            "Target (<50)": 50,
        }
        colors = ["#6b7787", "#6b7787", "#4ade80", "#38bdf8"]
        fig = go.Figure(go.Bar(
            x=list(baselines.keys()),
            y=list(baselines.values()),
            marker_color=colors
        ))
        fig.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#c9d4e0", yaxis_title="Perplexity (lower is better)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Generation Samples")
        if "generation_samples" in results:
            for s in results["generation_samples"]:
                with st.expander(f"Prompt: {s['prompt']}"):
                    st.markdown(f"**Full output:**")
                    st.code(s["completion"], language=None)

    else:
        st.info("Run `python evaluation/evaluate.py` first to generate results.")
        st.code("python evaluation/evaluate.py", language="bash")

elif page == "Model Info":
    st.title("🔧 Model Architecture")
    st.markdown("""
### FinGPT-Tiny — GPT from Scratch

A decoder-only transformer trained entirely from scratch on 53.8MB of financial text.
No pretrained weights. No HuggingFace trainer. Every component built in PyTorch.

```
Input: token ids (batch, seq_len)

  Token Embedding  (16,000 → 256)      # learned from scratch
  ↓
  Transformer Block × 6
    ├─ RMSNorm(256)                      # more stable than LayerNorm
    ├─ CausalSelfAttention(8 heads)
    │    ├─ Q, K, V projections
    │    ├─ Rotary Positional Encoding   # RoPE — same as Llama
    │    └─ Causal mask (no future peeking)
    ├─ Residual connection
    ├─ RMSNorm(256)
    └─ SwiGLU FFN (256 → 683 → 256)     # same activation as PaLM, Llama
  ↓
  Final RMSNorm
  ↓
  LM Head (256 → 16,000)                # weight-tied with embedding

Output: logits (batch, seq_len, 16,000)
```

### Training Details

| Config | Value |
|---|---|
| Corpus | 53.8MB financial text (SEC filings, Wikipedia finance, Reddit r/finance) |
| Tokenizer | BPE trained from scratch on corpus (16K vocab) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight decay=0.1) |
| LR Schedule | Linear warmup 200 steps → cosine decay |
| Batch size | 32 sequences × 256 tokens |
| Epochs | 5 (resumed from checkpoint) |
| Hardware | Apple M-series MPS |

### Key Design Choices

- **RoPE** instead of learned positional embeddings — better length generalisation
- **RMSNorm** instead of LayerNorm — faster, more stable for small models
- **SwiGLU** instead of ReLU — same activation as Llama, PaLM
- **Weight tying** between token embedding and LM head — reduces parameters, improves training stability
- **BPE tokenizer** trained on financial corpus — learns domain-specific subwords (ĠEBITDA, ĠEPS, Ġguidance as single tokens)
""")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Parameters", "8,819,456")
        st.metric("Embedding Parameters", "4,505,600")
        st.metric("Transformer Parameters", "4,313,856")
    with col2:
        st.metric("Vocabulary Size", "16,000")
        st.metric("Context Length", "256 tokens")
        st.metric("Model Size (fp32)", "~35 MB")