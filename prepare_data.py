"""Prepare The Movies Dataset files for RecoLab.

The original Kaggle files are kept unchanged. This script creates a compact,
normalized data directory that the application can consume.
"""

import argparse
import ast
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path("data/Movies dataset")
DEFAULT_OUTPUT = Path("data/processed")


def parse_json_list(value: object, key: str = "name") -> list[str]:
    """Extract names from Kaggle's Python-literal JSON-like columns."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        items = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    return [
        str(item[key]).strip()
        for item in items
        if isinstance(item, dict) and item.get(key)
    ]


def extract_director(value: object) -> str:
    """Extract the director name from a credits crew column."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        crew = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return ""
    if not isinstance(crew, list):
        return ""
    for member in crew:
        if (
            isinstance(member, dict)
            and member.get("job") == "Director"
            and member.get("name")
        ):
            return str(member["name"]).strip()
    return ""


def prepare_data(source_dir: Path, output_dir: Path) -> None:
    """Normalize the Kaggle dataset into files used by RecoLab."""
    metadata = pd.read_csv(
        source_dir / "movies_metadata.csv", low_memory=False
    )
    links = pd.read_csv(source_dir / "links.csv")
    keywords = pd.read_csv(source_dir / "keywords.csv")
    credits = pd.read_csv(source_dir / "credits.csv")
    ratings = pd.read_csv(source_dir / "ratings_small.csv")

    metadata["tmdbId"] = pd.to_numeric(metadata["id"], errors="coerce")
    metadata = metadata.dropna(subset=["tmdbId"]).copy()
    metadata["tmdbId"] = metadata["tmdbId"].astype("int64")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype("int64")

    movies = metadata.merge(links[["movieId", "tmdbId"]], on="tmdbId", how="inner")
    movies["genres"] = movies["genres"].map(
        lambda value: "|".join(parse_json_list(value))
    )
    movies["genres"] = movies["genres"].replace("", "(no genres listed)")
    movies["title"] = movies["title"].fillna("").astype(str).str.strip()
    movies = movies[movies["title"].ne("")].copy()

    credits["movieId"] = pd.to_numeric(credits["id"], errors="coerce")
    credits = credits.dropna(subset=["movieId"]).copy()
    credits["movieId"] = credits["movieId"].astype("int64")
    credits["cast_names"] = credits["cast"].map(
        lambda value: "|".join(parse_json_list(value)[:5])
    )
    credits["director"] = credits["crew"].map(extract_director)
    movies = movies.merge(
        credits[["movieId", "cast_names", "director"]],
        on="movieId",
        how="left",
    )

    keywords["movieId"] = pd.to_numeric(keywords["id"], errors="coerce")
    keywords = keywords.dropna(subset=["movieId"]).copy()
    keywords["movieId"] = keywords["movieId"].astype("int64")
    keywords["tag"] = keywords["keywords"].map(
        lambda value: "|".join(parse_json_list(value))
    )
    keyword_tags = keywords[["movieId", "tag"]]
    keyword_tags = keyword_tags.assign(tag=keyword_tags["tag"].str.split("|")).explode(
        "tag"
    )
    keyword_tags["tag"] = keyword_tags["tag"].str.strip()
    keyword_tags = keyword_tags[keyword_tags["tag"].ne("")]

    output_dir.mkdir(parents=True, exist_ok=True)
    movie_columns = [
        "movieId",
        "title",
        "genres",
        "overview",
        "poster_path",
        "release_date",
        "popularity",
        "vote_average",
        "vote_count",
        "cast_names",
        "director",
    ]
    movies[movie_columns].drop_duplicates("movieId").to_csv(
        output_dir / "movies.csv", index=False
    )
    keyword_tags.drop_duplicates().to_csv(output_dir / "tags.csv", index=False)
    ratings[["userId", "movieId", "rating", "timestamp"]].to_csv(
        output_dir / "ratings.csv", index=False
    )
    links[["movieId", "tmdbId", "imdbId"]].drop_duplicates("movieId").to_csv(
        output_dir / "links.csv", index=False
    )

    print(f"Prepared {len(movies):,} movies")
    print(f"Prepared {len(ratings):,} ratings")
    print(f"Prepared {len(keyword_tags):,} keyword tags")
    print(f"Output directory: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare_data(args.source, args.output)


if __name__ == "__main__":
    main()
