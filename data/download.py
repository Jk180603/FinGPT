"""
Get 50MB+ of financial text for FinGPT-Tiny pretraining
Primary source: The Pile - Financial subset (guaranteed large)
"""
from datasets import load_dataset
import pandas as pd
import os
import requests
import time

os.makedirs("data/raw", exist_ok=True)
all_texts = []

# Source 1: RedditFinance - large financial discussion corpus
print("Source 1: Reddit Finance posts...")
try:
    ds = load_dataset("winddude/reddit_finance_43_250k", split="train", streaming=True)
    count = 0
    for item in ds:
        text = item.get("body", item.get("selftext", item.get("title", "")))
        if isinstance(text, str) and len(text) > 100:
            all_texts.append(text[:3000])
            count += 1
            if count >= 30000:
                break
        if count % 5000 == 0 and count > 0:
            print(f"  Reddit Finance: {count} posts")
    print(f"  Total Reddit Finance: {count} posts")
except Exception as e:
    print(f"  Reddit Finance failed: {e}")

# Source 2: Financial news large corpus
print("Source 2: Financial news corpus...")
try:
    ds2 = load_dataset("ashraq/financial-news-articles", split="train", streaming=True)
    count = 0
    for item in ds2:
        text = item.get("text", item.get("article", ""))
        if isinstance(text, str) and len(text) > 200:
            all_texts.append(text[:5000])
            count += 1
            if count >= 20000:
                break
    print(f"  Financial news: {count} articles")
except Exception as e:
    print(f"  Financial news: {e}")

# Source 3: SEC EDGAR filings - 10-K annual reports
print("Source 3: SEC 10-K annual report excerpts...")
try:
    ds3 = load_dataset("JanosAudran/financial-reports-sec", split="train", streaming=True)
    count = 0
    for item in ds3:
        text = item.get("text", item.get("content", ""))
        if isinstance(text, str) and len(text) > 200:
            all_texts.append(text[:5000])
            count += 1
            if count >= 20000:
                break
        if count % 5000 == 0 and count > 0:
            print(f"  SEC 10-K: {count} docs")
    print(f"  Total SEC 10-K: {count} docs")
except Exception as e:
    print(f"  SEC 10-K failed: {e}")

# Source 4: FiQA financial Q&A
print("Source 4: FiQA dataset...")
try:
    ds4 = load_dataset("pauri32/fiqa-2018")
    for split in ds4:
        for item in ds4[split]:
            for key in ["question", "answer", "sentence"]:
                if key in item and isinstance(item[key], str) and len(item[key]) > 50:
                    all_texts.append(item[key])
    print(f"  FiQA added, total so far: {len(all_texts)}")
except Exception as e:
    print(f"  FiQA: {e}")

# Source 5: Wikipedia finance (extended list)
print("Source 5: Wikipedia finance articles (extended)...")
try:
    topics = [
        "Earnings_per_share", "Revenue", "Cash_flow", "Balance_sheet",
        "Income_statement", "Financial_statement", "Stock_market",
        "Investment_banking", "Hedge_fund", "Private_equity",
        "Venture_capital", "Initial_public_offering", "Bond_market",
        "Derivative_(finance)", "Option_(finance)", "Futures_contract",
        "Interest_rate", "Inflation", "Monetary_policy", "Federal_Reserve",
        "Goldman_Sachs", "JPMorgan_Chase", "BlackRock", "Berkshire_Hathaway",
        "Warren_Buffett", "Benjamin_Graham", "Value_investing",
        "Technical_analysis", "Fundamental_analysis", "Price-to-earnings_ratio",
        "Dividend", "Stock_split", "Share_repurchase", "Market_capitalization",
        "Enterprise_value", "EBITDA", "Net_income", "Gross_profit",
        "Operating_leverage", "Financial_leverage", "Working_capital",
        "Accounts_receivable", "Accounts_payable", "Inventory",
        "Depreciation", "Amortization", "Capital_expenditure",
        "Return_on_equity", "Return_on_assets", "Return_on_investment",
        "Discounted_cash_flow", "Net_present_value", "Internal_rate_of_return",
        "Capital_asset_pricing_model", "Beta_(finance)", "Alpha_(finance)",
        "Sharpe_ratio", "Portfolio_theory", "Efficient_market_hypothesis",
        "Behavioral_finance", "Market_anomaly", "Short_selling",
        "Margin_trading", "Securities_fraud", "Insider_trading",
        "Ponzi_scheme", "Enron_scandal", "Wirecard_scandal",
        "2008_financial_crisis", "Dot-com_bubble", "Subprime_mortgage_crisis",
        "Quantitative_easing", "Yield_curve", "Credit_default_swap",
        "Collateralized_debt_obligation", "Mortgage-backed_security",
        "Exchange-traded_fund", "Mutual_fund", "Index_fund",
        "Venture_debt", "Mezzanine_financing", "Convertible_bond",
        "Credit_rating", "Moody's", "Standard_&_Poor's", "Fitch_Ratings",
    ]

    wiki_count = 0
    for topic in topics:
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&titles={topic}&prop=extracts&explaintext=1&format=json"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                pages = resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    text = page.get("extract", "")
                    if len(text) > 500:
                        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 100]
                        all_texts.extend(paragraphs)
                        wiki_count += len(paragraphs)
            time.sleep(0.1)
        except:
            continue
    print(f"  Wikipedia: {wiki_count} paragraphs added")
except Exception as e:
    print(f"  Wikipedia: {e}")

# Clean and deduplicate
print("\nCleaning and deduplicating...")
seen = set()
clean_texts = []
for t in all_texts:
    if not isinstance(t, str):
        continue
    t = t.strip()
    if len(t) < 50:
        continue
    key = t[:100]
    if key in seen:
        continue
    seen.add(key)
    clean_texts.append(t)

df = pd.DataFrame({"text": clean_texts})
df.to_csv("data/raw/financial_corpus.csv", index=False)

total_chars = df["text"].str.len().sum()
print(f"\nCorpus statistics:")
print(f"  Total documents: {len(df):,}")
print(f"  Total characters: {total_chars:,}")
print(f"  Approx size: {total_chars / (1024*1024):.1f} MB")
print(f"  Avg document length: {df['text'].str.len().mean():.0f} chars")
print(f"\nSaved to data/raw/financial_corpus.csv")