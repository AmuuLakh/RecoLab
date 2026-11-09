"""Content-based movie recommendations using a compatible movie catalogue."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MOVIE_COLUMNS = ("movieId", "title", "genres")
RATING_COLUMNS = ("movieId", "rating")
TAG_COLUMNS = ("movieId", "tag")


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


def _validate_columns(data: pd.DataFrame, required_columns: tuple[str, ...], name: str) -> None:
    missing_columns = set(required_columns).difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{name} data is missing required columns: {missing}")


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Load ratings used to calculate catalogue quality signals."""
    ratings = pd.read_csv(path)
    _validate_columns(ratings, RATING_COLUMNS, "Ratings")
    ratings = ratings[list(RATING_COLUMNS)].dropna().copy()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])
    return ratings


def load_tags(path: str | Path) -> pd.DataFrame:
    """Load user tags used to add semantic similarity."""
    tags = pd.read_csv(path)
    _validate_columns(tags, TAG_COLUMNS, "Tags")
    tags = tags[list(TAG_COLUMNS)].dropna().copy()
    tags["tag"] = tags["tag"].astype(str).str.strip()
    return tags[tags["tag"].ne("")]


class MovieRecommender:
    """Recommend movies using genres, tags, and community ratings."""

    def __init__(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame | None = None,
        tags: pd.DataFrame | None = None,
    ) -> None:
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
        self._tag_matrix = None
        self._tag_lookup: dict[int, set[str]] = {}
        self._rating_stats = pd.DataFrame(
            0.0,
            index=self.movies.index,
            columns=["average_rating", "rating_count", "rating_score"],
        )

        if tags is not None:
            _validate_columns(tags, TAG_COLUMNS, "Tags")
            tag_data = tags[list(TAG_COLUMNS)].dropna().copy()
            tag_data["tag"] = tag_data["tag"].astype(str).str.strip()
            tag_data = tag_data[tag_data["tag"].ne("")]
            tag_text = tag_data.groupby("movieId")["tag"].apply(
                lambda values: " ".join(values)
            )
            tag_documents = self.movies["movieId"].map(tag_text).fillna("")
            self._tag_lookup = {
                movie_id: {tag.lower() for tag in group["tag"]}
                for movie_id, group in tag_data.groupby("movieId")
            }
            if tag_documents.str.strip().any():
                self._tag_matrix = TfidfVectorizer(
                    lowercase=True, stop_words="english"
                ).fit_transform(tag_documents)

        if ratings is not None:
            _validate_columns(ratings, RATING_COLUMNS, "Ratings")
            rating_data = ratings[list(RATING_COLUMNS)].dropna().copy()
            rating_data["rating"] = pd.to_numeric(
                rating_data["rating"], errors="coerce"
            )
            rating_data = rating_data.dropna(subset=["rating"])
            self._rating_stats = (
                self.movies[["movieId"]]
                .merge(
                    rating_data.groupby("movieId")["rating"].agg(
                        average_rating="mean", rating_count="count"
                    ),
                    on="movieId",
                    how="left",
                )
                .fillna(0.0)
            )
            self._rating_stats["rating_score"] = self._calculate_rating_scores()

        self._title_to_indices = (
            self.movies.reset_index()
            .groupby("title")["index"]
            .apply(list)
            .to_dict()
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path = "data/movies.csv",
        ratings_path: str | Path | None = "data/ratings.csv",
        tags_path: str | Path | None = "data/tags.csv",
    ) -> "MovieRecommender":
        """Create a recommender from catalogue, ratings, and tag CSV files."""
        ratings = load_ratings(ratings_path) if ratings_path is not None else None
        tags = load_tags(tags_path) if tags_path is not None else None
        return cls(load_movies(path), ratings=ratings, tags=tags)

    def _calculate_rating_scores(self) -> pd.Series:
        """Combine average rating and rating volume into a 0-to-1 signal."""
        average = self._rating_stats["average_rating"]
        count = np.log1p(self._rating_stats["rating_count"])
        average_score = ((average - 0.5) / 4.5).clip(0, 1)
        count_score = (count / count.max()) if count.max() else count
        return (0.7 * average_score + 0.3 * count_score).clip(0, 1)

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
        """Return hybrid recommendations with an explanation for each result."""
        if number_of_recommendations < 1:
            raise ValueError("number_of_recommendations must be at least 1")

        matching_indices = self._title_to_indices.get(movie_title)
        if matching_indices is None:
            raise ValueError(f"Movie not found: {movie_title}")

        selected_index = matching_indices[0]
        scores = cosine_similarity(
            self._genre_matrix[selected_index], self._genre_matrix
        ).ravel()
        genre_scores = scores.copy()

        if self._tag_matrix is not None:
            tag_scores = cosine_similarity(
                self._tag_matrix[selected_index], self._tag_matrix
            ).ravel()
        else:
            tag_scores = np.zeros(len(self.movies))

        rating_scores = self._rating_stats["rating_score"].to_numpy()
        scores = 0.55 * genre_scores + 0.25 * tag_scores + 0.20 * rating_scores
        scores[selected_index] = -1

        recommendation_count = min(number_of_recommendations, len(self.movies) - 1)
        recommended_indices = scores.argsort()[::-1][:recommendation_count]
        recommendations = self.movies.iloc[recommended_indices].copy()
        recommendations.insert(
            len(recommendations.columns), "similarity", scores[recommended_indices]
        )
        recommendations["genre_similarity"] = genre_scores[recommended_indices]
        recommendations["tag_similarity"] = tag_scores[recommended_indices]
        recommendations["average_rating"] = self._rating_stats.loc[
            recommended_indices, "average_rating"
        ].to_numpy()
        recommendations["rating_count"] = self._rating_stats.loc[
            recommended_indices, "rating_count"
        ].astype(int).to_numpy()
        recommendations["explanation"] = [
            self._explain_recommendation(selected_index, index)
            for index in recommended_indices
        ]
        return recommendations.reset_index(drop=True)

    def _explain_recommendation(self, selected_index: int, recommended_index: int) -> str:
        """Explain the strongest shared signals behind one recommendation."""
        selected_genres = set(self.movies.iloc[selected_index]["genres"].split("|"))
        recommended_genres = set(self.movies.iloc[recommended_index]["genres"].split("|"))
        shared_genres = sorted(selected_genres & recommended_genres)
        movie_id = self.movies.iloc[recommended_index]["movieId"]
        selected_id = self.movies.iloc[selected_index]["movieId"]
        shared_tags = sorted(
            self._tag_lookup.get(movie_id, set())
            & self._tag_lookup.get(selected_id, set())
        )
        reasons = []
        if shared_genres:
            reasons.append(f"shares {', '.join(shared_genres[:3])}")
        if shared_tags:
            reasons.append(f"matches tags like {', '.join(shared_tags[:2])}")
        rating = self._rating_stats.iloc[recommended_index]
        if rating["rating_count"] > 0:
            reasons.append(
                f"rated {rating['average_rating']:.1f}/5 by {int(rating['rating_count']):,} viewers"
            )
        return "Recommended because it " + "; ".join(reasons) + "."
