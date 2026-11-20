"""Robust readers for the raw Kaggle files. Train.csv is standard CSV. Test.csv has NO quoting, so commas inside
titles and author lists split fields; we rebuild each row from its fixed anchors: id (first), doi (last) and the
(venue, 4-digit year) pair, which occurs exactly once. Authors are re-joined with ", " as in train.csv."""
import csv, re, pandas as pd
VENUES = {"cav", "lics", "kr", "lpnmr", "iclp"}


def read_train(path):
    return pd.read_csv(path)


def read_test(path):
    rows = list(csv.reader(open(path, encoding="utf-8", newline="")))
    assert rows[0] == ["id", "title", "venue", "year", "authors", "doi"], rows[0]
    out = []
    for t in rows[1:]:
        if not t: continue
        cand = [i for i in range(2, len(t) - 2) if t[i].strip().lower() in VENUES and re.fullmatch(r"\d{4}", t[i + 1].strip())]
        assert len(cand) == 1, (t[:3], cand)
        i = cand[0]; au = [a.strip() for a in t[i + 2:-1] if a.strip()]
        out.append({"id": int(t[0]), "title": ", ".join(p.strip() for p in t[1:i]), "venue": t[i].strip(), "year": int(t[i + 1]),
                    "authors": ", ".join(au) if au else None, "doi": t[-1]})
    return pd.DataFrame(out)
