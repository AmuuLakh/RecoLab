import pandas as pd

# Load the datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Inspect the movie dataset
print("Movies Dataset:")
print(movies.head())

print("\nMovie dataset shape:", movies.shape)
print("\nMovie columns:", movies.columns.tolist())

# Inspect the ratings dataset
print("\nRatings Dataset:")
print(ratings.head())

print("\nRatings dataset shape:", ratings.shape)
print("\nRatings columns:",ratings.columns.tolist())


# 1. Find the five most common genres
all_genres = movies["genres"].str.split("|").explode()

print("\nTop 5 most common genres:")
print(all_genres.value_counts().head(5))

# 2. Find the highest - rated movies with at least 50 ratings.
movie_stats = ratings.groupby("movieId")["rating"].agg(
    average_rating = "mean",
    number_of_ratings = "count"
)

#Keep movies with at least 50 ratings
popular_movies = movie_stats[
    movie_stats["number_of_ratings"] >= 50]

#Sort by average rating, highest first
top_movies = popular_movies.sort_values(
    "average_rating",
    ascending=False
).head(5)

# Add movie titles to the results
top_movies = top_movies.reset_index().merge(
    movies[["movieId","title"]],
    on="movieId"
)

print("\nTop 5 movies by avergae rating(Minimum 50 ratings):")
print(
    top_movies[
        ["title","average_rating","number_of_ratings"]
    ]
)

# 3.Count unique users
unique_users = ratings["userId"].nunique()

print("\nNumber of unique users:", unique_users)


