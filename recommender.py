"""Content-based movie recommendations using MovieLens genres."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """Recommend movies with similar genre profiles."""

    def __init__(self, movies: pd.DataFrame) -> None:
        required_columns = {"movieId", "title", "genres"}
        missing_columns = required_columns.difference(movies.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Movies data is missing required columns: {missing}")

        self.movies = movies.reset_index(drop=True).copy()
        self.movies["genres"] = self.movies["genres"].fillna("(no genres listed)")

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
        """Create a recommender from a MovieLens movies CSV file."""
        return cls(pd.read_csv(path))

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
