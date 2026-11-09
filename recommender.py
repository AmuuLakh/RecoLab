"""Content-based movie recommendations using a compatible movie catalogue."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MOVIE_COLUMNS = ("movieId", "title", "genres")


def load_movies(path: str | Path) -> pd.DataFrame:
    """Load and validate a movie catalogue from a CSV file.

    Compatible catalogues must provide ``movieId``, ``title``, and ``genres``.
    Other columns are preserved so they can be used by future recommenders.
    """
    movies = pd.read_csv(path)
    missing_columns = set(MOVIE_COLUMNS).difference(movies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Movies data is missing required columns: {missing}")
    if movies.empty:
        raise ValueError(f"Movies data is empty: {path}")
    if movies["movieId"].duplicated().any():
        raise ValueError("Movies data must contain unique movieId values")

    movies = movies.copy()
    movies["title"] = movies["title"].fillna("").astype(str).str.strip()
    movies["genres"] = movies["genres"].fillna("(no genres listed)").astype(str)
    if movies["title"].eq("").any():
        raise ValueError("Movies data contains a movie with an empty title")
    return movies


class MovieRecommender:
    """Recommend movies with similar genre profiles."""

    def __init__(self, movies: pd.DataFrame) -> None:
        missing_columns = set(MOVIE_COLUMNS).difference(movies.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Movies data is missing required columns: {missing}")
        if movies.empty:
            raise ValueError("Movies data cannot be empty")
        if movies["movieId"].duplicated().any():
            raise ValueError("Movies data must contain unique movieId values")

        self.movies = movies.reset_index(drop=True).copy()
        self.movies["title"] = self.movies["title"].fillna("").astype(str).str.strip()
        self.movies["genres"] = self.movies["genres"].fillna(
            "(no genres listed)"
        ).astype(str)
        if self.movies["title"].eq("").any():
            raise ValueError("Movies data contains a movie with an empty title")

        self._vectorizer = CountVectorizer(
            tokenizer=lambda value: value.split("|"),
            token_pattern=None,
            lowercase=False,
        )
        self._genre_matrix = self._vectorizer.fit_transform(self.movies["genres"])
        self._title_to_indices = (
            self.movies.reset_index()
            .groupby("title")["index"]
            .apply(list)
            .to_dict()
        )

    @classmethod
    def from_csv(cls, path: str | Path = "data/movies.csv") -> "MovieRecommender":
        """Create a recommender from any compatible movie catalogue CSV."""
        return cls(load_movies(path))

    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Find movie titles containing ``query`` without regular expressions."""
        query = query.strip()
        if not query:
            return self.movies.iloc[0:0][["movieId", "title", "genres"]]
        matches = self.movies[
            self.movies["title"].str.contains(query, case=False, na=False, regex=False)
        ]
        return matches[["movieId", "title", "genres"]].head(limit)

    def recommend_movies(
        self, movie_title: str, number_of_recommendations: int = 5
    ) -> pd.DataFrame:
        """Return movies with the most similar genre profiles."""
        if number_of_recommendations < 1:
            raise ValueError("number_of_recommendations must be at least 1")

        matching_indices = self._title_to_indices.get(movie_title)
        if matching_indices is None:
            raise ValueError(f"Movie not found: {movie_title}")

        selected_index = matching_indices[0]
        scores = cosine_similarity(
            self._genre_matrix[selected_index], self._genre_matrix
        ).ravel()
        scores[selected_index] = -1

        recommendation_count = min(number_of_recommendations, len(self.movies) - 1)
        recommended_indices = scores.argsort()[::-1][:recommendation_count]
        recommendations = self.movies.iloc[recommended_indices][
            ["movieId", "title", "genres"]
        ].copy()
        recommendations.insert(3, "similarity", scores[recommended_indices])
        return recommendations.reset_index(drop=True)
