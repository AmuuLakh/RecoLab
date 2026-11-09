"""Interactive command-line interface for RecoLab recommendations."""

import argparse

from recommender import MovieRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Find similar movies in a CSV catalogue.")
    parser.add_argument(
        "--movies",
        default="data/movies.csv",
        help="Path to a compatible movie catalogue CSV file.",
    )
    args = parser.parse_args()
    recommender = MovieRecommender.from_csv(args.movies)

    print("RecoLab Movie Recommender")
    print("Type a movie title, or press Enter to exit.")

    while True:
        query = input("\nSearch for a movie: ").strip()
        if not query:
            print("Goodbye!")
            return

        matches = recommender.search_movies(query)
        if matches.empty:
            print(f"No movies found matching '{query}'.")
            continue

        if len(matches) > 1 or matches.iloc[0]["title"].lower() != query.lower():
            print("\nMatching movies:")
            for index, title in enumerate(matches["title"], start=1):
                print(f"{index}. {title}")
            selection = input("Choose a movie number (or press Enter to cancel): ").strip()
            if not selection:
                continue
            try:
                selected_position = int(selection) - 1
                movie_title = matches.iloc[selected_position]["title"]
            except (ValueError, IndexError):
                print("Please choose a valid movie number.")
                continue
        else:
            movie_title = matches.iloc[0]["title"]

        recommendations = recommender.recommend_movies(movie_title)
        print(f"\nMovies similar to {movie_title}:")
        print(recommendations[["title", "genres", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    main()
