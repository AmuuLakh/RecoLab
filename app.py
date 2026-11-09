"""RecoLab's Streamlit web interface."""

from html import escape
from pathlib import Path

import streamlit as st

from recommender import MovieRecommender


DATASET_PATH = Path("data/processed/movies.csv")

st.set_page_config(
    page_title="RecoLab | Find your next favorite",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_recommender() -> MovieRecommender:
    """Load the recommendation engine once for the current app session."""
    return MovieRecommender.from_csv(
        DATASET_PATH,
        ratings_path=Path("data/processed/ratings.csv"),
        tags_path=Path("data/processed/tags.csv"),
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #f7f7fb;
            --muted: #a7a5b8;
            --panel: rgba(31, 28, 51, 0.78);
            --line: rgba(255, 255, 255, 0.1);
            --violet: #9a7cff;
            --pink: #f27bc4;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(116, 83, 255, .26), transparent 28rem),
                radial-gradient(circle at 92% 12%, rgba(242, 123, 196, .16), transparent 25rem),
                #0c0b14;
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 2rem; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; }
        .brand { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; letter-spacing: -.03em; }
        .brand span { color: var(--pink); }
        .eyebrow { color: var(--pink); font-size: .75rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
        .hero { padding: 3.5rem 0 2.2rem; max-width: 760px; }
        .hero h1 { font-size: clamp(2.6rem, 6vw, 5rem); letter-spacing: -.065em; line-height: .98; margin: .7rem 0 1.1rem; }
        .hero h1 span { background: linear-gradient(100deg, #f8f6ff, #bba8ff 55%, #f27bc4); -webkit-background-clip: text; color: transparent; }
        .hero p { color: var(--muted); font-size: 1.08rem; line-height: 1.65; max-width: 620px; }
        .search-label { color: var(--muted); font-size: .82rem; font-weight: 600; margin-bottom: .4rem; }
        div[data-baseweb="input"] { background: rgba(255,255,255,.07); border: 1px solid var(--line); border-radius: 12px; }
        div[data-baseweb="input"] input { color: var(--ink); }
        .stButton > button { border: 0; border-radius: 10px; background: linear-gradient(100deg, var(--violet), var(--pink)); color: white; font-weight: 700; min-height: 42px; }
        .stButton > button:hover { border: 0; color: white; filter: brightness(1.12); }
        div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; }
        .section-title { margin: 2.8rem 0 1rem; font-size: 1.35rem; font-weight: 700; }
        .movie-card { background: linear-gradient(150deg, rgba(48, 43, 76, .9), rgba(24, 22, 38, .95)); border: 1px solid var(--line); border-radius: 15px; min-height: 175px; padding: 1.15rem; }
        .movie-number { color: var(--pink); font-size: .75rem; font-weight: 700; }
        .movie-title { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.08rem; font-weight: 600; line-height: 1.2; margin: .5rem 0 .7rem; }
        .movie-genres { color: var(--muted); font-size: .82rem; line-height: 1.5; }
        .score { color: #bba8ff; font-size: .78rem; font-weight: 700; margin-top: 1rem; }
        .note { color: var(--muted); font-size: .9rem; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_movie_card(movie, position: int) -> None:
    genres = " · ".join(
        genre for genre in str(movie["genres"]).split("|") if genre != "(no genres listed)"
    )
    similarity = f"{float(movie['similarity']) * 100:.0f}% match"
    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-number">0{position}</div>
            <div class="movie-title">{escape(str(movie["title"]))}</div>
            <div class="movie-genres">{escape(genres or "Genre not available")}</div>
                <div class="score">{similarity} · hybrid match</div>
                <div class="movie-genres" style="margin-top:.7rem">{escape(str(movie["explanation"]))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    recommender = load_recommender()

    st.markdown('<div class="brand">Reco<span>Lab</span></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Your personal movie compass</div>
            <h1>Find a film<br><span>worth remembering.</span></h1>
            <p>Tell us what you are in the mood for. RecoLab blends genres, community tags, and ratings to find films with a similar DNA.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("movie_search"):
        st.markdown('<div class="search-label">SEARCH THE CATALOGUE</div>', unsafe_allow_html=True)
        query = st.text_input(
            "Movie title",
            placeholder="Try “Toy Story”, “The Matrix”, or “Inception”",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Search movies", use_container_width=True)

    if submitted:
        matches = recommender.search_movies(query)
        st.session_state["matches"] = matches
        st.session_state.pop("recommendations", None)
        if matches.empty:
            st.warning("No movies found. Try a different title or a shorter search.")

    matches = st.session_state.get("matches")
    if matches is None or matches.empty:
        st.markdown(
            '<div class="section-title">Start with a favorite</div><div class="note">Search above to unlock a personalized row of recommendations.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="section-title">Choose your starting movie</div>', unsafe_allow_html=True)
    selected_title = st.selectbox(
        "Matching movies",
        matches["title"].tolist(),
        label_visibility="collapsed",
    )
    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=12,
        value=6,
        label_visibility="collapsed",
    )
    if st.button("Discover similar movies", use_container_width=True):
        st.session_state["recommendations"] = recommender.recommend_movies(
            selected_title, recommendation_count
        )

    recommendations = st.session_state.get("recommendations")
    if recommendations is None:
        return

    st.markdown(
        f'<div class="section-title">Because you liked <span style="color:#f27bc4">{escape(selected_title)}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="note" style="margin-bottom:1.2rem">A curated selection powered by genres, community tags, and ratings.</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    for position, (_, movie) in enumerate(recommendations.iterrows(), start=1):
        with columns[(position - 1) % 3]:
            render_movie_card(movie, position)


if __name__ == "__main__":
    main()
