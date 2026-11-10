import json
import re
import time
import urllib.request
import pandas as pd
from pathlib import Path

DATA_DIR = Path("source-code/data")
CACHE_FILE = DATA_DIR / "abstracts_cache.json"

def main():
    print("=== FETCHING SEMANTIC SCHOLAR ABSTRACTS KEYED BY PAPER ID ===")
    
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"Initial cache entries: {len(cache)}")

    # Load datasets
    train_df = pd.read_csv("source-code/data/train.csv")
    test_df = pd.read_csv("source-code/data/test.csv")
    combined = pd.concat([train_df[["id", "doi", "title"]], test_df[["id", "doi", "title"]]], ignore_index=True)

    s2_pattern = re.compile(r"semanticscholar\.org/paper/([a-f0-9]{40})", re.IGNORECASE)
    
    # Map paper_id (hex) -> list of dataset paper IDs (str)
    targets = {}
    already_in_cache_by_doi = 0
    for _, row in combined.iterrows():
        p_id = str(row["id"])
        doi_val = str(row["doi"]) if pd.notna(row["doi"]) else ""
        
        # Check if already cached under p_id
        if p_id in cache and cache[p_id] and len(cache[p_id]) > 30:
            continue
            
        # Check if accidentally cached under doi_val from previous run
        if doi_val in cache and cache[doi_val] and len(cache[doi_val]) > 30:
            cache[p_id] = cache[doi_val]
            already_in_cache_by_doi += 1
            continue

        match = s2_pattern.search(doi_val)
        if match:
            hex_id = match.group(1).lower()
            if hex_id not in targets:
                targets[hex_id] = []
            targets[hex_id].append(p_id)

    print(f"Recovered {already_in_cache_by_doi} abstracts directly from previous doi-keyed cache.")
    print(f"Remaining Semantic Scholar hex IDs to query: {len(targets)}")

    hex_list = list(targets.keys())
    batch_size = 100
    retrieved_count = 0
    url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract"

    if hex_list:
        for i in range(0, len(hex_list), batch_size):
            batch = hex_list[i : i + batch_size]
            print(f"Querying batch {i//batch_size + 1}/{(len(hex_list)+batch_size-1)//batch_size} ({len(batch)} IDs)...")
            payload = json.dumps({"ids": batch}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "AntigravityDataMining/1.0 (academic research)"
                }
            )

            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    results = json.loads(resp.read().decode())
                    for hex_id, res in zip(batch, results):
                        for p_id in targets[hex_id]:
                            if res and res.get("abstract"):
                                cache[p_id] = res["abstract"].strip()
                                retrieved_count += 1
                            else:
                                if p_id not in cache:
                                    cache[p_id] = ""
            except Exception as e:
                print(f"  Error on batch: {e}")
            
            time.sleep(1.2)

    print(f"\nRetrieved {retrieved_count} new abstracts via API.")
    
    # Save cache
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # Export CSVs with proper mapping
    train_df["abstract"] = train_df["id"].astype(str).map(cache).fillna("")
    test_df["abstract"]  = test_df["id"].astype(str).map(cache).fillna("")

    train_valid = (train_df["abstract"].str.len() > 30).sum()
    test_valid  = (test_df["abstract"].str.len() > 30).sum()

    print(f"\n[FINAL ABSTRACT COVERAGE]")
    print(f"  Train: {train_valid}/{len(train_df)} ({train_valid/len(train_df)*100:.2f}%)")
    print(f"  Test:  {test_valid}/{len(test_df)} ({test_valid/len(test_df)*100:.2f}%)")

    train_df.to_csv(DATA_DIR / "train_with_abstracts.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_with_abstracts.csv", index=False)
    print("[SUCCESS] train_with_abstracts.csv and test_with_abstracts.csv successfully updated!")

if __name__ == "__main__":
    main()
