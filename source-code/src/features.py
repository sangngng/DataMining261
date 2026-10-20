import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import Tuple, Dict, Any

class FeaturePipeline:
    """
    Feature extraction pipeline combining:
      1. Text features: TF-IDF on paper title
      2. Metadata features: year, venue frequency, author statistics, doi presence
      3. Clustering features: K-Means topic clusters on title TF-IDF
    """
    def __init__(self, tfidf_max_features: int = 2500, n_clusters: int = 8, random_state: int = 42):
        self.tfidf_max_features = tfidf_max_features
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english',
            min_df=2
        )
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.venue_counts = {}
        self.median_year = 2020.0
        self.fitted = False

    def _extract_meta_features(self, df: pd.DataFrame, is_train: bool = True) -> np.ndarray:
        # Title length features
        title_series = df['title'].fillna('')
        char_len = title_series.apply(len).values.reshape(-1, 1)
        word_count = title_series.apply(lambda s: len(s.split())).values.reshape(-1, 1)

        # Year feature
        year_series = pd.to_numeric(df['year'], errors='coerce')
        if is_train:
            self.median_year = float(year_series.median())
        year_clean = year_series.fillna(self.median_year).values.reshape(-1, 1)
        year_norm = (year_clean - 2000.0) / 30.0

        # DOI presence
        has_doi = (~df['doi'].isna()).astype(float).values.reshape(-1, 1)

        # Author count
        author_series = df['authors'].fillna('')
        author_count = author_series.apply(lambda s: len(s.split(',')) if len(s.strip()) > 0 else 0).values.reshape(-1, 1)

        # Venue frequency encoding
        venue_series = df['venue'].fillna('UNKNOWN').astype(str)
        if is_train:
            self.venue_counts = venue_series.value_counts().to_dict()
        venue_freq = venue_series.map(self.venue_counts).fillna(0.0).values.reshape(-1, 1)
        venue_freq_log = np.log1p(venue_freq)

        meta_matrix = np.hstack([char_len, word_count, year_norm, has_doi, author_count, venue_freq_log])
        return meta_matrix

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit feature extractors on training data and return feature matrix."""
        titles = df['title'].fillna('').astype(str)
        tfidf_feats = self.tfidf.fit_transform(titles).toarray()

        # Fit KMeans clustering (a core taught technique in Data Mining)
        cluster_distances = self.kmeans.fit_transform(tfidf_feats)

        # Metadata features
        meta_feats = self._extract_meta_features(df, is_train=True)

        self.fitted = True
        return np.hstack([tfidf_feats, cluster_distances, meta_feats])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform unseen data (e.g. test set or validation fold)."""
        assert self.fitted, "Pipeline must be fitted first before transform!"
        titles = df['title'].fillna('').astype(str)
        tfidf_feats = self.tfidf.transform(titles).toarray()

        cluster_distances = self.kmeans.transform(tfidf_feats)
        meta_feats = self._extract_meta_features(df, is_train=False)

        return np.hstack([tfidf_feats, cluster_distances, meta_feats])
