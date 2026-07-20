"""
FinGPT-Tiny Pretraining Loop - with resume from checkpoint
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow
import os
import sys
import json
import math
import time
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

sys.path.insert(0, ".")
from model.fingpt import FinGPT


class FinancialTextDataset(Dataset):
    def __init__(self, token_ids: list, seq_len: int = 256):
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)
        print(f"Dataset: {len(self.data):,} tokens, {len(self):,} sequences of length {seq_len}")

    def __len__(self):
        return (len(self.data) - self.seq_len) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


def tokenize_corpus(tokenizer, texts, cache_path="data/processed/tokens.npy"):
    if os.path.exists(cache_path):
        print(f"Loading cached tokens from {cache_path}...")
        return np.load(cache_path).tolist()

    print(f"Tokenizing {len(texts):,} documents...")
    os.makedirs("data/processed", exist_ok=True)

    all_tokens = []
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    for i, text in enumerate(texts):
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue
        encoded = tokenizer.encode(text.strip())
        all_tokens.append(bos_id)
        all_tokens.extend(encoded.ids)
        all_tokens.append(eos_id)

        if (i + 1) % 5000 == 0:
            print(f"  Tokenized {i+1:,}/{len(texts):,} docs, {len(all_tokens):,} tokens so far")

    np.save(cache_path, np.array(all_tokens, dtype=np.int32))
    print(f"Saved {len(all_tokens):,} tokens to {cache_path}")
    return all_tokens


def find_latest_checkpoint():
    """Find the highest step checkpoint available"""
    ckpt_dir = "model/checkpoints"
    if not os.path.exists(ckpt_dir):
        return None, 0
    
    step_files = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("step_") and f.endswith(".pt"):
            try:
                step = int(f.replace("step_", "").replace(".pt", ""))
                step_files.append((step, os.path.join(ckpt_dir, f)))
            except:
                continue
    
    if not step_files:
        return None, 0
    
    step_files.sort(reverse=True)
    latest_step, latest_path = step_files[0]
    return latest_path, latest_step


def train():
    CONFIG = {
        "vocab_size": 16000,
        "dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "max_seq_len": 256,
        "dropout": 0.1,
        "seq_len": 256,
        "batch_size": 32,
        "lr": 3e-4,
        "min_lr": 3e-5,
        "epochs": 5,
        "warmup_steps": 200,
        "grad_clip": 1.0,
        "eval_interval": 200,
        "save_interval": 500,
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file("tokenizer/financial_bpe.json")
    print(f"Vocabulary: {tokenizer.get_vocab_size():,} tokens")

    print("Loading financial corpus...")
    df = pd.read_csv("data/raw/financial_corpus.csv")
    texts = df["text"].dropna().tolist()
    print(f"Corpus: {len(texts):,} documents")

    all_tokens = tokenize_corpus(tokenizer, texts)
    print(f"Total tokens: {len(all_tokens):,}")

    split = int(len(all_tokens) * 0.95)
    train_tokens = all_tokens[:split]
    val_tokens = all_tokens[split:]

    train_dataset = FinancialTextDataset(train_tokens, CONFIG["seq_len"])
    val_dataset = FinancialTextDataset(val_tokens, CONFIG["seq_len"])

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=0
    )

    print(f"\nTrain batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")

    model = FinGPT(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # ── RESUME FROM CHECKPOINT ────────────────────────────────
    resume_path, resume_step = find_latest_checkpoint()
    if resume_path:
        print(f"\nResuming from {resume_path} (step {resume_step:,})...")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Resumed successfully at step {resume_step:,}")
    else:
        resume_step = 0
        print("\nNo checkpoint found, starting from scratch...")
    # ──────────────────────────────────────────────────────────

    total_steps = len(train_loader) * CONFIG["epochs"]

    def get_lr(step):
        if step < CONFIG["warmup_steps"]:
            return CONFIG["lr"] * step / max(CONFIG["warmup_steps"], 1)
        progress = (step - CONFIG["warmup_steps"]) / max(total_steps - CONFIG["warmup_steps"], 1)
        return CONFIG["min_lr"] + 0.5 * (CONFIG["lr"] - CONFIG["min_lr"]) * (1 + math.cos(math.pi * progress))

    mlflow.set_experiment("fingpt-tiny-pretraining")
    os.makedirs("model/checkpoints", exist_ok=True)

    with mlflow.start_run(run_name=f"fingpt-tiny-resume-{resume_step}"):
        mlflow.log_params(CONFIG)
        mlflow.log_param("resumed_from_step", resume_step)
        mlflow.log_param("model_parameters", model.count_parameters())

        global_step = resume_step
        best_val_loss = float("inf")

        for epoch in range(1, CONFIG["epochs"] + 1):
            model.train()
            epoch_losses = []
            epoch_start = time.time()

            for batch_idx, (x, y) in enumerate(train_loader):

                # ── SKIP ALREADY TRAINED STEPS ────────────────
                current_step = (epoch - 1) * len(train_loader) + batch_idx
                if current_step < resume_step:
                    global_step = max(global_step, current_step + 1)
                    continue
                # ──────────────────────────────────────────────

                x, y = x.to(device), y.to(device)

                lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.zero_grad()
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                optimizer.step()

                epoch_losses.append(loss.item())
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = np.mean(epoch_losses[-50:])
                    perplexity = math.exp(min(avg_loss, 20))
                    elapsed = time.time() - epoch_start
                    tokens_per_sec = len(epoch_losses) * CONFIG["batch_size"] * CONFIG["seq_len"] / max(elapsed, 1)
                    print(
                        f"Epoch {epoch}/{CONFIG['epochs']} | "
                        f"Step {global_step:,} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"PPL: {perplexity:.1f} | "
                        f"LR: {lr:.2e} | "
                        f"{tokens_per_sec:.0f} tok/s"
                    )
                    mlflow.log_metrics({
                        "train_loss": avg_loss,
                        "train_perplexity": perplexity,
                        "learning_rate": lr,
                    }, step=global_step)

                if global_step % CONFIG["eval_interval"] == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for val_x, val_y in val_loader:
                            val_x, val_y = val_x.to(device), val_y.to(device)
                            _, val_loss = model(val_x, val_y)
                            val_losses.append(val_loss.item())

                    avg_val_loss = np.mean(val_losses)
                    val_ppl = math.exp(min(avg_val_loss, 20))
                    print(f"\n{'='*60}")
                    print(f"Validation Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:.1f}")

                    mlflow.log_metrics({
                        "val_loss": avg_val_loss,
                        "val_perplexity": val_ppl,
                    }, step=global_step)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), "model/checkpoints/best_model.pt")
                        print(f"New best model saved! Val PPL: {val_ppl:.1f}")
                    print(f"{'='*60}\n")
                    model.train()

                if global_step % CONFIG["save_interval"] == 0:
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "global_step": global_step,
                        "config": CONFIG,
                    }, f"model/checkpoints/step_{global_step}.pt")
                    print(f"Checkpoint saved: step_{global_step}.pt")

        results = {
            "best_val_loss": best_val_loss,
            "best_val_perplexity": math.exp(min(best_val_loss, 20)),
            "total_steps": global_step,
            "total_tokens_trained": global_step * CONFIG["batch_size"] * CONFIG["seq_len"],
            "model_parameters": model.count_parameters(),
            "resumed_from": resume_step,
            "architecture": "GPT with RoPE + RMSNorm + SwiGLU",
        }

        with open("model/pretrain_results.json", "w") as f:
            json.dump(results, f, indent=2)

        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "best_val_perplexity": results["best_val_perplexity"]
        })

        print("\n" + "="*60)
        print("PRETRAINING COMPLETE")
        print("="*60)
        print(f"Best Val Loss:        {best_val_loss:.4f}")
        print(f"Best Val Perplexity:  {results['best_val_perplexity']:.1f}")
        print(f"Total steps:          {global_step:,}")
        print(f"Resumed from step:    {resume_step:,}")
        print(f"Model parameters:     {model.count_parameters():,}")
        print("="*60)


if __name__ == "__main__": 
    train()