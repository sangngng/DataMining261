"""Feature spaces. Unsupervised representations are built once on ALL available text (train + test titles/abstracts),
identically in cross-validation and in the final fit (declared unsupervised use of unlabeled test text)."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from .data import text_of, SEED

class Space:
    """Unsupervised representations built once on ALL available text (train+test, declared)."""
    def __init__(self, tr, te, Etr, Ete):
        self.df = tr.reset_index(drop=True)._append(te.reset_index(drop=True), ignore_index=True)
        self.ntr = len(tr)
        txt = text_of(self.df)
        self.Xw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit_transform(txt)
        self.Xc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True).fit_transform(self.df["title"].fillna(""))
        self.E = np.vstack([Etr, Ete]).astype(np.float32)
        ven = self.df["venue"].astype(str)
        self.vcodes = ven.astype("category").cat.codes.values
        V = sparse.csr_matrix(np.eye(self.vcodes.max() + 1)[self.vcodes])
        yr = self.df["year"].clip(2016, 2025).values
        Y = sparse.csr_matrix(np.eye(10)[yr - 2016] * 0.5)
        self.VY = sparse.hstack([V, Y]).tocsr()
        self.vy_code = self.vcodes * 10 + (yr - 2016)
        self.Xw_v = sparse.hstack([self.Xw, self.VY]).tocsr()
        self.Xc_v = sparse.hstack([self.Xc, self.VY]).tocsr()
        self.E_v = np.hstack([self.E, self.VY.toarray()]).astype(np.float32)
        from sklearn.cluster import KMeans
        from sklearn.decomposition import TruncatedSVD
        self.km12 = KMeans(12, n_init=5, random_state=SEED).fit_predict(self.E)
        self.km40 = KMeans(40, n_init=5, random_state=SEED).fit_predict(self.E)
        self.svd = TruncatedSVD(48, random_state=SEED).fit_transform(self.Xw)
        # items for Apriori-style rules: frequent title/abstract unigrams (df>=12), venue, year-bin
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(binary=True, min_df=12, max_df=0.5, stop_words="english", token_pattern=r"[A-Za-z][A-Za-z\-]{2,}")
        W = cv.fit_transform(txt)
        ybin = sparse.csr_matrix(np.eye(3)[np.digitize(self.df["year"].values, [2019, 2022])])
        self.items = sparse.hstack([W, V, ybin]).tocsr().astype(np.float32)
        t = self.df["title"].fillna("")
        self.tl = t.str.split().str.len().values; self.tc = t.str.len().values
        self.tag = t.str.contains(r"\((?:short|invited|extended|tutorial|talk|abstract|demo|poster|tool|survey|keynote)", case=False, regex=True).values.astype(int)
        self.meta_kw = t.str.contains(r"proceedings|preface|workshop|volume|editorial|invited|keynote|tutorial|panel", case=False, regex=True).values.astype(int)
        vg = np.where(np.isin(self.df["venue"].astype(str), ["cav", "lics"]), self.df["venue"].astype(str).map({"cav": 0, "lics": 1}).fillna(0), 2).astype(int)
        Vg = np.eye(3)[np.where(self.df["venue"].astype(str) == "cav", 0, np.where(self.df["venue"].astype(str) == "lics", 1, 2))]
        self.E_vi = np.hstack([self.E] + [self.E * Vg[:, [j]] for j in range(3)]).astype(np.float32)
        self.Xt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit_transform(self.df["title"].fillna(""))
        self.Xt_v = sparse.hstack([self.Xt, self.VY]).tocsr()
        ab = self.df["abstract"].fillna("") if "abstract" in self.df else self.df["title"].fillna("")
        self.Xa = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=4, max_features=60000, sublinear_tf=True).fit_transform(ab)
        self.Xa_v = sparse.hstack([self.Xa, self.VY]).tocsr()
        self.Xwc = sparse.hstack([self.Xw, self.Xc, self.VY]).tocsr()
        self.authors = [set(a.split(", ")) - {""} if isinstance(a, str) else set() for a in self.df["authors"]]
        self.nauth = np.array([len(a) for a in self.authors])
        self.has_abs = self.df["abstract"].notna().values.astype(int) if "abstract" in self.df else np.zeros(len(self.df))
        self.year = self.df["year"].values
        self.global_ids = self.df["id"].values
