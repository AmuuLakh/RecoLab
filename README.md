# RecoLab

RecoLab is a movie recommendation system built with a pluggable movie
catalogue. It currently provides content-based recommendations by comparing
movie genre profiles. MovieLens is the initial dataset, not a permanent
limitation.

## Current features

- Search the MovieLens catalogue by title.
- Recommend similar movies using cosine similarity.
- Display a similarity score and the matching genres.
- Explore ratings, genres, and popular movies with `explore_data.py`.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python recommend.py
```

You can replace the catalogue without changing the recommendation engine:

```powershell
python recommend.py --movies path\to\your\movies.csv
```

The catalogue must contain these columns:

```text
movieId,title,genres
```

`movieId` must be unique. When merging datasets, normalize them to this schema
and deduplicate IDs before passing the merged CSV to RecoLab. Additional
columns are preserved for future features such as ratings, tags, posters, and
external links.

Enter part of a movie title, select the matching movie, and RecoLab will show
the closest genre-based recommendations. Press Enter at the search prompt to
exit.

## Project structure

```text
recolab/
├── data/                 # MovieLens data files
├── explore_data.py       # Dataset exploration script
├── recommender.py        # Reusable recommendation engine
├── recommend.py          # Command-line interface
└── requirements.txt      # Python dependencies
```

## Roadmap

- Add a Streamlit interface for a portfolio-ready demo.
- Combine genres, tags, and ratings into a hybrid recommender.
- Add offline evaluation with Precision@K and Recall@K.
- Add recommendation explanations and automated tests.

## Dataset

RecoLab uses the MovieLens `ml-latest-small` dataset from GroupLens. See
[`data/README.txt`](data/README.txt) for the dataset description, license, and
citation requirements.
