"""
fetch_abstracts.py: Multi-Source Robust Abstract Retrieval for Train & Test Papers
Retrieves abstracts via OpenAlex Batch DOI queries, OpenAlex Title Search, and CrossRef fallback.
Includes caching, HTML entity unescaping, and rate-limit compliance.

Run with:
/Users/sanggnguyenn/Documents/HCMUT/DataMining/.venv/bin/python source-code/src/fetch_abstracts.py
"""
import os
import sys
import re
import html
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

DATA_DIR = Path("source-code/data")
CACHE_FILE = DATA_DIR / "abstracts_cache.json"
HEADERS = {"User-Agent": "mailto:sang.nguyen.hcmut@gmail.com (Academic Research Data Mining Project)"}

def clean_title_str(t):
    if pd.isna(t):
        return ""
    t_clean = html.unescape(str(t)).strip()
    t_clean = re.sub(r"\s+", " ", t_clean)
    # Strip leading/trailing quotation marks or periods
    t_clean = re.sub(r"^[\"\'\.\s]+|[\"\'\.\s]+$", "", t_clean)
    return t_clean

def extract_standard_doi(doi_str):
    if pd.isna(doi_str):
        return None
    d = str(doi_str).strip()
    # Match standard DOI pattern 10.xxxx/...
    m = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", d)
    if m:
        # Strip trailing punctuation
        doi = m.group(1).rstrip(".)/,;")
        return doi.lower()
    return None

def reconstruct_inverted_index(inv):
    if not inv:
        return None
    words = sorted([(pos, w) for w, pos_list in inv.items() for pos in pos_list])
    return " ".join([w for _, w in words]).strip()

def load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def fetch_openalex_doi_batch(doi_list):
    """
    Batch query OpenAlex for a list of DOIs (up to 50 at a time).
    Returns a dict mapping doi.lower() -> abstract_text.
    """
    if not doi_list:
        return {}
    doi_urls = [f"https://doi.org/{d}" for d in doi_list]
    doi_filter = "|".join(doi_urls)
    url = f"https://api.openalex.org/works?filter=doi:{urllib.parse.quote(doi_filter)}&per-page={len(doi_list)}"
    
    results = {}
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            for item in data.get("results", []):
                item_doi = extract_standard_doi(item.get("doi", ""))
                inv = item.get("abstract_inverted_index")
                abs_text = reconstruct_inverted_index(inv)
                if item_doi and abs_text:
                    results[item_doi] = abs_text
    except Exception as e:
        print(f"  [Batch DOI Warning] {e}")
    return results

def fetch_single_title_openalex(title):
    clean_t = clean_title_str(title)
    if len(clean_t) < 5:
        return None
    t_enc = urllib.parse.quote(clean_t)
    url = f"https://api.openalex.org/works?filter=title.search:{t_enc}&per-page=3"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            for item in data.get("results", []):
                # Verify title similarity loosely
                ret_title = clean_title_str(item.get("title", ""))
                # Compare first few words
                w_orig = clean_t.lower().split()[:4]
                w_ret = ret_title.lower().split()[:4]
                overlap = len(set(w_orig).intersection(set(w_ret)))
                if overlap >= min(2, len(w_orig)):
                    inv = item.get("abstract_inverted_index")
                    abs_text = reconstruct_inverted_index(inv)
                    if abs_text and len(abs_text) > 30:
                        return abs_text
    except Exception:
        pass
    return None

def fetch_single_title_crossref(title):
    clean_t = clean_title_str(title)
    if len(clean_t) < 5:
        return None
    t_enc = urllib.parse.quote(clean_t)
    url = f"https://api.crossref.org/works?query.title={t_enc}&rows=3"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            items = data.get("message", {}).get("items", [])
            for item in items:
                raw_abs = item.get("abstract")
                if raw_abs:
                    # Strip jats xml tags
                    clean_abs = re.sub(r"<[^>]+>", " ", raw_abs).strip()
                    clean_abs = re.sub(r"\s+", " ", clean_abs)
                    if len(clean_abs) > 30:
                        return clean_abs
    except Exception:
        pass
    return None

def main():
    print("="*75)
    print("=== STARTING FULL ACADEMIC ABSTRACT RETRIEVAL (OPENALEX + CROSSREF) ===")
    print("="*75)
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")
    
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    total_papers = len(all_df)
    print(f"Total papers to enrich: {total_papers} (Train: {len(train_df)}, Test: {len(test_df)})")
    
    cache = load_cache()
    print(f"Loaded existing cache with {len(cache)} entries.")
    
    # Extract DOIs and clean titles
    all_df["clean_title"] = all_df["title"].apply(clean_title_str)
    all_df["standard_doi"] = all_df["doi"].apply(extract_standard_doi)
    
    # Phase 1: OpenAlex Batch DOI queries for un-cached papers with DOIs
    uncached_dois = []
    for _, r in all_df.iterrows():
        p_id = str(r["id"])
        doi = r["standard_doi"]
        if p_id not in cache and doi:
            uncached_dois.append((p_id, doi))
            
    print(f"\n[PHASE 1] Batch DOI Retrieval for {len(uncached_dois)} papers...")
    batch_size = 40
    newly_found_doi = 0
    for i in range(0, len(uncached_dois), batch_size):
        chunk = uncached_dois[i:i+batch_size]
        dois_in_chunk = [doi for _, doi in chunk]
        batch_results = fetch_openalex_doi_batch(dois_in_chunk)
        
        for p_id, doi in chunk:
            if doi in batch_results:
                cache[p_id] = batch_results[doi]
                newly_found_doi += 1
                
        time.sleep(0.3) # Polite rate limit
        if (i // batch_size) % 5 == 0:
            print(f"  Processed {min(i+batch_size, len(uncached_dois))}/{len(uncached_dois)} DOIs (Found: {newly_found_doi})...")
            save_cache(cache)
            
    save_cache(cache)
    print(f"[PHASE 1 COMPLETE] Retrieved {newly_found_doi} abstracts via batch DOI. Total cached: {len(cache)}.")
    
    # Phase 2: Title Search (OpenAlex with CrossRef fallback) for remaining papers
    missing_items = []
    for _, r in all_df.iterrows():
        p_id = str(r["id"])
        if p_id not in cache or not cache[p_id] or len(cache[p_id]) < 20:
            missing_items.append((p_id, r["clean_title"]))
            
    print(f"\n[PHASE 2] Multi-Threaded Title Search for {len(missing_items)} remaining papers...")
    
    def worker_fetch(item):
        p_id, title = item
        # 1. Try OpenAlex Title
        abs_text = fetch_single_title_openalex(title)
        # 2. Try CrossRef Title
        if not abs_text:
            abs_text = fetch_single_title_crossref(title)
        return p_id, abs_text
        
    found_phase2 = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(worker_fetch, it): it for it in missing_items}
        for count, fut in enumerate(as_completed(futures), 1):
            p_id, abs_text = fut.result()
            if abs_text:
                cache[p_id] = abs_text
                found_phase2 += 1
            else:
                # Mark as empty string to avoid re-querying repeatedly
                cache[p_id] = ""
                
            if count % 100 == 0 or count == len(missing_items):
                print(f"  Processed {count}/{len(missing_items)} titles (Found in Phase 2: {found_phase2})...")
                save_cache(cache)
                
    save_cache(cache)
    print(f"[PHASE 2 COMPLETE] Retrieved {found_phase2} abstracts via Title Search.")
    
    # Final Summary & Export
    train_df["abstract"] = train_df["id"].astype(str).map(cache).fillna("")
    test_df["abstract"]  = test_df["id"].astype(str).map(cache).fillna("")
    
    train_cov = (train_df["abstract"].str.len() > 30).sum() / len(train_df)
    test_cov  = (test_df["abstract"].str.len() > 30).sum() / len(test_df)
    
    print("\n" + "="*75)
    print("=== FINAL ENRICHMENT SUMMARY ===")
    print("="*75)
    print(f"Train Papers with Abstract (>30 chars): {(train_df['abstract'].str.len() > 30).sum()}/{len(train_df)} ({train_cov*100:.2f}%)")
    print(f"Test Papers with Abstract (>30 chars):  {(test_df['abstract'].str.len() > 30).sum()}/{len(test_df)} ({test_cov*100:.2f}%)")
    print(f"Mean Abstract Word Count (Train): {train_df['abstract'].apply(lambda x: len(x.split())).mean():.1f} words")
    print(f"Mean Abstract Word Count (Test):  {test_df['abstract'].apply(lambda x: len(x.split())).mean():.1f} words")
    
    train_out = DATA_DIR / "train_with_abstracts.csv"
    test_out  = DATA_DIR / "test_with_abstracts.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"\n[SUCCESS] Saved enriched datasets:")
    print(f"  -> {train_out}")
    print(f"  -> {test_out}")

if __name__ == "__main__":
    main()
