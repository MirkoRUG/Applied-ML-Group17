import streamlit as st
import requests
import pandas as pd
from movie_score_predictor.models.ensemble_model \
    import MovieScoreEnsemblePredictor

# Constants
API_KEY = "dcbe7f561670aeffbe72a416d3c16a08"
BASE_URL = "https://api.themoviedb.org/3"

st.set_page_config(page_title="Movie Score Predictor", page_icon="🎬")


@st.cache_resource
def load_model():
    """Load and train the ensemble model."""
    try:
        model = MovieScoreEnsemblePredictor(n_models=3)
        df = pd.read_csv("data.csv")
        model.train(df)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


def api_request(url, params):
    try:
        return requests.get(url, params=params).json()
    except requests.RequestException:
        return None


def get_movie_features(movie_data, credits_data):
    """Extract required features from TMDB data."""
    if not movie_data or not credits_data:
        return None

    # Basic info
    release_date = movie_data.get('release_date', '')
    year = int(release_date[:4]) if release_date else None

    # Crew info
    crew = credits_data.get('crew', [])
    director = next(
        (p['name'] for p in crew if p.get('job') == 'Director'),
        'Unknown'
    )
    writer = next(
        (p['name'] for p in crew if p.get('job') in ['Writer', 'Screenplay']),
        'Unknown'
    )

    return {
        'name': movie_data.get('title', 'Unknown'),
        'rating': 'PG-13',  # Default, TMDb does not contain MPA rating
        'genre': movie_data.get('genres', [{}])[0].get('name', 'Unknown'),
        'year': year,
        'released': release_date,
        'director': director,
        'writer': writer,
        'star': credits_data.get('cast', [{}])[0].get('name', 'Unknown'),
        'country': movie_data.get('production_countries',
                                  [{}])[0].get('name', 'Unknown'),
        'budget': movie_data.get('budget') if movie_data.get('budget', 0) > 0
                  else None,
        'company': movie_data.get('production_companies',
                                  [{}])[0].get('name', 'Unknown'),
        'runtime': movie_data.get('runtime', None)
    }


def show_prediction(prediction, confidence):
    """Display prediction results."""
    uncertainty = "Low 🟢" if confidence < 0.2 else "Medium 🟡"\
        if confidence < 0.5 else "High 🔴"

    cols = st.columns(3)
    cols[0].metric("Predicted Score", f"{prediction:.2f}/10")
    cols[1].metric("Confidence", f"±{confidence:.2f}")
    cols[2].metric("Uncertainty", uncertainty)

    st.progress(min(int(prediction * 10), 100))
    st.caption(f"Predicted range: {max(0, prediction - confidence):.1f} - \
               {min(10, prediction + confidence):.1f}")


def show_movie_details(features):
    """Show all the features fetched from TMDb"""
    with st.expander("Full Movie Details", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"**{features['name']}** ({features['year']})")
            st.markdown(f"**Genre:** {features['genre']}")
            st.markdown(f"**Released:** {features['released']}")
            st.markdown(f"**Runtime:** {features['runtime']} minutes"
                        if features['runtime'] else "**Runtime:** Unknown")
            st.markdown(f"**Rating:** {features['rating']}")

        with col2:
            st.subheader("Production Details")
            st.markdown(f"**Director:** {features['director']}")
            st.markdown(f"**Writer:** {features['writer']}")
            st.markdown(f"**Star:** {features['star']}")
            st.markdown(f"**Production Company:** {features['company']}")

        # Bottom section
        st.markdown("---")
        st.markdown(f"**Country:** {features['country']}")
        if features['budget']:
            st.markdown(f"**Budget:** ${features['budget']:,}")
        else:
            st.markdown("**Budget:** Unknown")


def main():
    """Main app interface."""
    st.title("Movie Score Predictor")
    st.write("Search for any movie to get a score prediction")

    model = load_model()
    if not model:
        st.stop()

    # Search interface
    query = st.text_input("Search movies:", placeholder="e.g., Inception")

    if not query:
        return

    # Search TMDB
    with st.spinner("Searching..."):
        results = api_request(
            f"{BASE_URL}/search/movie",
            {"api_key": API_KEY, "query": query}
        )

    if not results or not results.get('results'):
        st.warning("No movies found")
        return

    # Movie selection
    movies = results['results'][:5]  # Show top 5
    selection = st.selectbox(
        "Select movie:",
        movies,
        format_func=lambda m: f"{m['title']} \
            ({m.get('release_date', '?')[:4]})"
    )

    if not st.button("Predict Score"):
        return

    # Fetch details
    with st.spinner("Fetching details..."):
        movie_id = selection['id']
        movie_data = api_request(
            f"{BASE_URL}/movie/{movie_id}",
            {"api_key": API_KEY}
        )
        credits_data = api_request(
            f"{BASE_URL}/movie/{movie_id}/credits",
            {"api_key": API_KEY}
        )

    if not movie_data or not credits_data:
        st.error("Failed to fetch details")
        return

    # Show info and predict
    features = get_movie_features(movie_data, credits_data)
    if not features:
        st.error("Failed to process features")
        return

    show_movie_details(features)

    with st.spinner("Predicting..."):
        try:
            pred, conf = model.predict(features,
                                       return_uncertainty=True)  # type: ignore
            show_prediction(pred, conf)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
