# RecoLab

RecoLab is a movie recommendation system built with the MovieLens dataset. It
currently provides content-based recommendations by comparing movie genre
profiles.

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
