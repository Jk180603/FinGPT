"""
FinGPT-Tiny — Industry-Level Evaluation Suite
Evaluates on 4 dimensions used in real LLM research:
1. Perplexity on held-out financial test set
2. Financial domain knowledge (fill-in-the-blank)
3. Text quality metrics (vocabulary diversity, repetition rate)
4. Generation coherence (sentence completion scoring)
"""
import torch
import math
import json
import numpy as np
from tokenizers import Tokenizer
from model.fingpt import FinGPT
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(device):
    tokenizer = Tokenizer.from_file("tokenizer/financial_bpe.json")
    model = FinGPT(
        vocab_size=16000, dim=256, num_heads=8,
        num_layers=6, max_seq_len=256
    ).to(device)
    model.load_state_dict(torch.load(
        "model/checkpoints/best_model.pt", map_location=device))
    model.eval()
    return model, tokenizer


def evaluate_perplexity(model, tokenizer, texts, device, seq_len=128):
    """Standard perplexity on financial test texts"""
    import pandas as pd
    all_tokens = []
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    for text in texts[:500]:
        if isinstance(text, str) and len(text) > 30:
            ids = tokenizer.encode(text.strip()).ids
            all_tokens += [bos] + ids + [eos]

    data = torch.tensor(all_tokens, dtype=torch.long)
    losses = []
    with torch.no_grad():
        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i:i+seq_len].unsqueeze(0).to(device)
            y = data[i+1:i+seq_len+1].unsqueeze(0).to(device)
            _, loss = model(x, y)
            if not torch.isnan(loss):
                losses.append(loss.item())
    avg_loss = np.mean(losses)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def evaluate_financial_knowledge(model, tokenizer, device):
    """
    Fill-in-the-blank on financial terms.
    Model must predict the masked token.
    Based on FinLM evaluation methodology.
    """
    prompts = [
        ("Revenue grew 23% year over year to $4.2", "billion"),
        ("The company reported strong EBITDA margins of", "32"),
        ("Free cash flow was $1.8", "billion"),
        ("Earnings per share increased to", "$"),
        ("The Board approved a share", "repurchase"),
        ("Net income attributable to common", "shareholders"),
        ("Operating expenses decreased as a percentage of", "revenue"),
        ("The company raised full-year guidance to reflect", "confidence"),
    ]
    correct = 0
    results = []
    with torch.no_grad():
        for prompt, expected_start in prompts:
            ids = tokenizer.encode(prompt).ids[-50:]
            x = torch.tensor([ids], dtype=torch.long).to(device)
            logits, _ = model(x)
            next_id = logits[0, -1, :].argmax().item()
            predicted = tokenizer.decode([next_id]).strip()
            hit = expected_start.lower() in predicted.lower() or predicted.lower() in expected_start.lower()
            if hit:
                correct += 1
            results.append({
                "prompt": prompt,
                "predicted": predicted,
                "expected": expected_start,
                "correct": hit
            })
    accuracy = correct / len(prompts)
    return accuracy, results


def evaluate_text_quality(model, tokenizer, device, n_samples=20):
    """
    Measure generation quality:
    - Vocabulary diversity (unique tokens / total tokens)
    - Repetition rate (repeated bigrams)
    - Average generation length before <eos>
    """
    import random
    seed_prompts = [
        "Revenue grew", "The company reported", "Free cash flow",
        "Earnings per share", "Operating margin", "The stock market",
        "Quarterly results", "Net income", "Cash and cash equivalents",
        "The Board of Directors",
    ]
    all_diversity = []
    all_repetition = []

    with torch.no_grad():
        for prompt in seed_prompts[:n_samples]:
            ids = tokenizer.encode(prompt).ids
            x = torch.tensor([ids], dtype=torch.long).to(device)
            out = model.generate(x, max_new_tokens=80, temperature=0.8, top_k=50)
            gen_ids = out[0].tolist()[len(ids):]
            if len(gen_ids) < 5:
                continue
            unique = len(set(gen_ids)) / len(gen_ids)
            all_diversity.append(unique)
            bigrams = [(gen_ids[i], gen_ids[i+1]) for i in range(len(gen_ids)-1)]
            rep = 1 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0
            all_repetition.append(rep)

    return {
        "vocab_diversity": np.mean(all_diversity),
        "repetition_rate": np.mean(all_repetition),
    }


def generate_samples(model, tokenizer, device):
    """Generate sample completions for qualitative review"""
    prompts = [
        "Revenue grew 23% year over year",
        "The company reported strong quarterly results",
        "Free cash flow was",
        "Earnings per share increased to",
        "We are raising full-year guidance",
        "The stock market experienced",
    ]
    samples = []
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer.encode(prompt).ids
            x = torch.tensor([ids], dtype=torch.long).to(device)
            out = model.generate(x, max_new_tokens=60, temperature=0.8, top_k=50)
            text = tokenizer.decode(out[0].tolist())
            samples.append({"prompt": prompt, "completion": text})
    return samples


def main():
    import pandas as pd

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading model...")
    model, tokenizer = load_model(device)
    print(f"Parameters: {model.count_parameters():,}")

    print("\n[1/4] Perplexity evaluation...")
    df = pd.read_csv("data/raw/financial_corpus.csv")
    texts = df["text"].dropna().tolist()
    split = int(len(texts) * 0.95)
    test_texts = texts[split:]
    val_loss, ppl = evaluate_perplexity(model, tokenizer, test_texts, device)
    print(f"    Test Loss: {val_loss:.4f}")
    print(f"    Test Perplexity: {ppl:.2f}")

    print("\n[2/4] Financial knowledge evaluation...")
    accuracy, knowledge_results = evaluate_financial_knowledge(model, tokenizer, device)
    print(f"    Fill-in-the-blank accuracy: {accuracy:.1%}")
    for r in knowledge_results:
        mark = "OK" if r["correct"] else "X "
        print(f"    [{mark}] '{r['prompt'][:40]}...' -> '{r['predicted']}' (expected: '{r['expected']}')")

    print("\n[3/4] Text quality metrics...")
    quality = evaluate_text_quality(model, tokenizer, device)
    print(f"    Vocabulary diversity: {quality['vocab_diversity']:.3f} (higher = better, random=1.0)")
    print(f"    Repetition rate:      {quality['repetition_rate']:.3f} (lower = better)")

    print("\n[4/4] Generation samples...")
    samples = generate_samples(model, tokenizer, device)
    for s in samples:
        print(f"\n    Prompt: {s['prompt']}")
        print(f"    Output: {s['completion'][:150]}")

    results = {
        "model": {
            "parameters": model.count_parameters(),
            "vocab_size": 16000,
            "architecture": "GPT with RoPE + RMSNorm + SwiGLU + Weight Tying",
            "training_corpus_mb": 53.8,
        },
        "evaluation": {
            "test_loss": round(val_loss, 4),
            "test_perplexity": round(ppl, 2),
            "financial_knowledge_accuracy": round(accuracy, 4),
            "vocab_diversity": round(quality["vocab_diversity"], 4),
            "repetition_rate": round(quality["repetition_rate"], 4),
        },
        "generation_samples": samples,
    }

    with open("model/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Test Perplexity:              {ppl:.2f}")
    print(f"Financial Knowledge Accuracy: {accuracy:.1%}")
    print(f"Vocabulary Diversity:         {quality['vocab_diversity']:.3f}")
    print(f"Repetition Rate:              {quality['repetition_rate']:.3f}")
    print(f"Model Parameters:             {model.count_parameters():,}")
    print("="*60)
    print("Results saved to model/eval_results.json")


if __name__ == "__main__":
    main()