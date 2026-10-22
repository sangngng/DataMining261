import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from typing import Dict, Any, List, Optional

VENUE_SMOOTHING_M = 5.0


def _split_authors(author_str: str) -> List[str]:
    """Split a raw 'authors' cell into a clean list of author name tokens."""
    if not isinstance(author_str, str) or len(author_str.strip()) == 0:
        return []
    return [a.strip() for a in author_str.split(',') if a.strip()]


class AuthorVenueEncoder:
    """
    Out-of-fold safe encoder for Author collaboration-count and Bayesian-smoothed
    Venue target encoding.

    IMPORTANT: `fit` must only ever be called on a train split (train fold or full train set).
    `transform` is then applied to both train and validation/test splits to avoid data leakage.
    """

    def __init__(self, venue_smoothing_m: float = VENUE_SMOOTHING_M):
        self.venue_smoothing_m = venue_smoothing_m
        self.author_count_map: Dict[str, int] = {}
        self.venue_smoothed_map: Dict[str, float] = {}
        self.global_mean: float = 0.0
        self.fitted: bool = False

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray) -> "AuthorVenueEncoder":
        y_train = np.asarray(y_train, dtype=float)
        self.global_mean = float(y_train.mean())

        # ---- Author paper-count only ----
        # NOTE: Author rating-history stats (max/mean/"is_preferred" rating per author)
        # were tested and REMOVED. With ~2500 train rows and mostly 1-2 papers per author,
        # those stats collapse to "this author's own current-row label" on train, which
        # the tree models latch onto as a near-perfect but totally non-generalizing signal
        # (measured OOF score 0.567-0.573 vs 0.613-0.615 without them). Only the paper
        # count (a structural, non-target-derived signal) is kept.
        author_counts: Dict[str, int] = {}
        author_series = df_train['authors'].fillna('')
        for authors_cell in author_series.values:
            for author in _split_authors(authors_cell):
                author_counts[author] = author_counts.get(author, 0) + 1
        self.author_count_map = author_counts

        # ---- Bayesian-smoothed Venue target encoding ----
        venue_series = df_train['venue'].fillna('UNKNOWN').astype(str)
        venue_df = pd.DataFrame({'venue': venue_series.values, 'y': y_train})
        venue_group = venue_df.groupby('venue')['y'].agg(['mean', 'count'])
        vm = self.venue_smoothing_m
        smoothed = (venue_group['count'] * venue_group['mean'] + vm * self.global_mean) / (venue_group['count'] + vm)
        self.venue_smoothed_map = smoothed.to_dict()

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Map author paper-count and Bayesian-smoothed venue stats onto `df`. Unseen
        authors/venues fall back to 0 / global_mean so no leakage from the current
        split ever occurs.
        """
        assert self.fitted, "AuthorVenueEncoder must be fit() on a train split before transform()!"

        author_series = df['authors'].fillna('')
        n = len(df)

        author_count = np.zeros(n, dtype=float)
        for i, authors_cell in enumerate(author_series.values):
            names = _split_authors(authors_cell)
            if len(names) == 0:
                continue
            count_vals = [self.author_count_map.get(a, 0) for a in names]
            author_count[i] = float(np.sum(count_vals))

        venue_series = df['venue'].fillna('UNKNOWN').astype(str)
        venue_smoothed = venue_series.map(self.venue_smoothed_map).fillna(self.global_mean).values.astype(float)

        return np.column_stack([
            author_count,        # author_paper_count (collaboration history size only)
            venue_smoothed,      # venue_bayesian_smoothed
        ])

    # Convenience metadata for callers that want to name the columns
    FEATURE_NAMES = [
        'author_paper_count',
        'venue_bayesian_smoothed',
    ]



def extract_features_v3(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    is_train: bool = True,
    kmeans_model: Optional[KMeans] = None,
    svd_model: Optional[TruncatedSVD] = None,
    pos_centroid: Optional[np.ndarray] = None,
    neg_centroid: Optional[np.ndarray] = None,
    author_venue_encoder: Optional[AuthorVenueEncoder] = None,
    y_train: Optional[np.ndarray] = None,
) -> "tuple[np.ndarray, Dict[str, Any]]":
    """
    V3 feature pipeline: SBERT embeddings + SVD(48) + K-Means(12) topic clusters +
    preference centroids + metadata + Out-of-fold-safe Author/Venue encoding.

    When `is_train=True`, all stateful transformers (SVD, K-Means, centroids,
    AuthorVenueEncoder) are fit on the given split only. When `is_train=False`,
    the fitted transformers passed in are reused (never refit), guaranteeing no leakage
    across CV folds.
    """
    # 1. Cosine similarity to Preference Centroids (positive: label>=4, negative: label==1)
    if is_train:
        pos_mask = (y_train >= 4)
        neg_mask = (y_train == 1)
        pos_centroid = embeddings[pos_mask].mean(axis=0, keepdims=True)
        neg_centroid = embeddings[neg_mask].mean(axis=0, keepdims=True)
        pos_centroid = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-9)
        neg_centroid = neg_centroid / (np.linalg.norm(neg_centroid) + 1e-9)

    sim_pos = np.dot(embeddings, pos_centroid.T)
    sim_neg = np.dot(embeddings, neg_centroid.T)
    sim_diff = sim_pos - sim_neg

    # 2. SVD Dimensionality Reduction on SBERT (top 48 latent semantic axes)
    if is_train:
        svd_model = TruncatedSVD(n_components=48, random_state=42)
        svd_feats = svd_model.fit_transform(embeddings)
    else:
        svd_feats = svd_model.transform(embeddings)

    # 3. K-Means Semantic Topic Clustering (K=12)
    if is_train:
        kmeans_model = KMeans(n_clusters=12, random_state=42, n_init=10)
        cluster_dists = kmeans_model.fit_transform(embeddings)
    else:
        cluster_dists = kmeans_model.transform(embeddings)

    # 4. Metadata features
    year_num = pd.to_numeric(df['year'], errors='coerce').fillna(2020.0).values.reshape(-1, 1)
    year_norm = (year_num - 2000.0) / 26.0
    paper_age = (2026.0 - year_num) / 10.0
    is_recent = (year_num >= 2024.0).astype(float)

    has_doi = (~df['doi'].isna()).astype(float).values.reshape(-1, 1)

    author_series = df['authors'].fillna('')
    author_count = author_series.apply(lambda s: len(s.split(',')) if len(s.strip()) > 0 else 0).values.reshape(-1, 1)
    is_single_author = (author_count == 1).astype(float)

    title_series = df['title'].fillna('')
    char_len = (title_series.apply(len).values.reshape(-1, 1) - 50.0) / 50.0
    word_count = (title_series.apply(lambda s: len(s.split())).values.reshape(-1, 1) - 10.0) / 10.0

    # 5. Out-of-fold-safe Author history + Bayesian-smoothed Venue encoding
    if is_train:
        author_venue_encoder = AuthorVenueEncoder()
        author_venue_encoder.fit(df, y_train)
    av_feats = author_venue_encoder.transform(df)  # (N, 5)

    # Assemble all features
    X = np.hstack([
        svd_feats,            # 48
        cluster_dists,        # 12
        sim_pos,               # 1
        sim_neg,               # 1
        sim_diff,              # 1
        year_norm,             # 1
        paper_age,             # 1
        is_recent,             # 1
        has_doi,               # 1
        author_count,          # 1
        is_single_author,      # 1
        char_len,              # 1
        word_count,            # 1
        av_feats,              # 2 (author_paper_count, venue_smoothed)
    ])

    cache = {
        'kmeans': kmeans_model,
        'svd': svd_model,
        'pos_centroid': pos_centroid,
        'neg_centroid': neg_centroid,
        'author_venue_encoder': author_venue_encoder,
    }
    return X, cache


FEATURE_COLUMN_NAMES = (
    [f'svd_{i}' for i in range(48)] +
    [f'kmeans_dist_{i}' for i in range(12)] +
    ['sim_pos', 'sim_neg', 'sim_diff', 'year_norm', 'paper_age', 'is_recent',
     'has_doi', 'author_count', 'is_single_author', 'char_len', 'word_count'] +
    AuthorVenueEncoder.FEATURE_NAMES
)
