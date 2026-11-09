"""RecoLab's Streamlit web interface."""

from html import escape
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

from recommender import MovieRecommender


DATASET_PATH = Path("data/processed/movies.csv")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

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
        .hero { padding: 4.5rem 0 2rem; max-width: 820px; text-align: center; margin: 0 auto; }
        .hero h1 { font-size: clamp(2.8rem, 7vw, 5.5rem); letter-spacing: -.075em; line-height: .96; margin: .7rem 0 1.1rem; }
        .hero h1 span { background: linear-gradient(100deg, #f8f6ff, #bba8ff 55%, #f27bc4); -webkit-background-clip: text; color: transparent; }
        .hero p { color: var(--muted); font-size: 1.08rem; line-height: 1.65; max-width: 620px; margin: 0 auto; }
        .search-label { color: var(--muted); font-size: .82rem; font-weight: 600; margin-bottom: .4rem; }
        div[data-baseweb="input"] { background: rgba(255,255,255,.07); border: 1px solid var(--line); border-radius: 12px; }
        div[data-baseweb="input"] input { color: var(--ink); }
        .stButton > button { border: 0; border-radius: 10px; background: linear-gradient(100deg, var(--violet), var(--pink)); color: white; font-weight: 700; min-height: 42px; }
        .stButton > button:hover { border: 0; color: white; filter: brightness(1.12); }
        div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; }
        .section-title { margin: 2.8rem 0 1rem; font-size: 1.35rem; font-weight: 700; }
        .movie-card { background: linear-gradient(150deg, rgba(48, 43, 76, .9), rgba(24, 22, 38, .95)); border: 1px solid var(--line); border-radius: 15px; min-height: 175px; padding: 1.15rem; }
        .poster-frame img { border-radius: 13px; box-shadow: 0 18px 34px rgba(0,0,0,.32); transition: transform .2s ease, box-shadow .2s ease; }
        .poster-frame img:hover { transform: translateY(-7px) scale(1.02); box-shadow: 0 26px 42px rgba(0,0,0,.46); }
        .poster-title { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: .98rem; font-weight: 600; line-height: 1.2; margin-top: .8rem; }
        .poster-meta { color: var(--muted); font-size: .76rem; margin-top: .35rem; }
        .rail-label { color: var(--muted); font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin: 1.8rem 0 1rem; }
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


def poster_url(movie) -> str | None:
    """Build a TMDB poster URL when the normalized catalogue has one."""
    poster_path = str(movie.get("poster_path", "") or "").strip()
    if not poster_path or poster_path == "nan":
        return None
    return f"{TMDB_IMAGE_BASE}{poster_path}"


@st.cache_data(show_spinner=False)
def load_poster(url: str) -> bytes | None:
    """Fetch a poster on the server so browser cross-origin rules do not block it."""
    request = Request(url, headers={"User-Agent": "RecoLab/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError):
        return None


def fallback_poster(movie) -> bytes:
    """Create a small branded poster when a remote image is unavailable."""
    title = str(movie["title"])[:42]
    genres = " · ".join(
        genre
        for genre in str(movie["genres"]).split("|")
        if genre != "(no genres listed)"
    )[:60]
    image = Image.new("RGB", (500, 750), "#17142b")
    draw = ImageDraw.Draw(image)
    draw.ellipse((250, -90, 610, 270), fill="#332b62")
    draw.ellipse((-170, 490, 260, 920), fill="#3b1f50")
    draw.text((42, 48), "RECOLAB", fill="#f27bc4", stroke_width=0)
    draw.text((42, 500), title, fill="#f7f7fb")
    draw.text((42, 548), genres, fill="#bba8ff")
    draw.text((42, 690), "POSTER UNAVAILABLE", fill="#a7a5b8")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def poster_bytes(movie) -> bytes:
    """Return a remote poster or an attractive local fallback."""
    poster = poster_url(movie)
    if poster:
        image = load_poster(poster)
        if image:
            return image
    return fallback_poster(movie)


def render_movie_card(movie, position: int) -> None:
    genres = " · ".join(
        genre for genre in str(movie["genres"]).split("|") if genre != "(no genres listed)"
    )
    similarity = f"{float(movie['similarity']) * 100:.0f}% match"
    st.image(poster_bytes(movie), use_container_width=True)
    st.markdown(
        f"""
        <div class="poster-title">{escape(str(movie["title"]))}</div>
        <div class="poster-meta">{escape(genres or "Genre not available")}</div>
        <div class="score">{similarity} · hybrid match</div>
        <div class="movie-genres" style="margin-top:.65rem">{escape(str(movie["explanation"]))}</div>
        """,
        unsafe_allow_html=True,
    )


def render_featured_rail(recommender: MovieRecommender) -> None:
    """Show a poster-led discovery rail before a search is performed."""
    featured = recommender.movies.copy()
    if "vote_count" in featured.columns:
        featured["vote_count"] = pd.to_numeric(featured["vote_count"], errors="coerce").fillna(0)
        featured = featured.sort_values("vote_count", ascending=False)
    featured = featured.head(6)
    st.markdown('<div class="rail-label">A few films to get you started</div>', unsafe_allow_html=True)
    columns = st.columns(6)
    for column, (_, movie) in zip(columns, featured.iterrows()):
        with column:
            st.image(poster_bytes(movie), use_container_width=True)
            st.markdown(
                f'<div class="poster-title">{escape(str(movie["title"]))}</div>',
                unsafe_allow_html=True,
            )


def main() -> None:
    inject_styles()
    recommender = load_recommender()

    st.markdown('<div class="brand">Reco<span>Lab</span></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI-powered movie discovery</div>
            <h1>Discover your next<br><span>favorite movie.</span></h1>
            <p>Search a catalogue of thousands of films and let RecoLab find your next great watch using genres, community tags, and ratings.</p>
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
        render_featured_rail(recommender)
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
