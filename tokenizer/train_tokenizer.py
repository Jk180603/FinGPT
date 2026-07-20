"""
Train BPE Tokenizer from scratch on financial text
"""
import pandas as pd
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

os.makedirs("tokenizer", exist_ok=True)

print("Loading financial corpus...")
df = pd.read_csv("data/raw/financial_corpus.csv")
texts = df["text"].dropna().tolist()
print(f"Loaded {len(texts):,} documents ({sum(len(t) for t in texts) / 1e6:.1f} MB)")

corpus_path = "data/raw/corpus.txt"
print("Writing corpus to text file...")
with open(corpus_path, "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text.strip() + "\n")
print(f"Corpus file: {os.path.getsize(corpus_path) / 1e6:.1f} MB")

# Build BPE tokenizer from scratch
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=16000,
    min_frequency=3,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"],
    show_progress=True,
)

print("\nTraining BPE tokenizer on 53.8MB of financial text...")
tokenizer.train([corpus_path], trainer)
tokenizer.save("tokenizer/financial_bpe.json")

print(f"\nVocabulary size: {tokenizer.get_vocab_size():,}")

# Test
test_sentences = [
    "Revenue grew 23% year over year to $4.2 billion.",
    "The company reported strong EBITDA margins of 32.4%.",
    "Free cash flow was $1.8 billion, up from $1.2 billion.",
    "We are raising full-year EPS guidance to $12.50 per share.",
    "Net income attributable to common shareholders was $892 million.",
]

print("\nTokenizer test:")
print("="*60)
for sentence in test_sentences:
    encoded = tokenizer.encode(sentence)
    print(f"\nInput:   {sentence}")
    print(f"Tokens:  {encoded.tokens}")
    print(f"Count:   {len(encoded.tokens)} tokens")

# Financial vocabulary learned
vocab = tokenizer.get_vocab()
fin_tokens = [t for t in vocab.keys() if any(w in t.lower() for w in
    ["revenue", "earn", "profit", "quarter", "share", "stock",
     "cash", "income", "margin", "growth", "guidance", "ebitda"])]
print(f"\nFinancial tokens learned: {len(fin_tokens)}")
print(f"Examples: {fin_tokens[:20]}")
print("\nTokenizer training complete!")