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
