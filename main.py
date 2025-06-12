import streamlit as st
from movie_score_predictor.models.catboost_model import get_model


# Model path
MODEL_PATH = "models/catboost_movie_model.cbm"

# App configuration
st.set_page_config(page_title="Movie Score Predictor", page_icon="🎬")


# Load your model
@st.cache_resource  # Cache the model load
def load_model():
    try:
        model = get_model(MODEL_PATH)
        st.success(" Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f" Failed to load model: {str(e)}")
        return None


def main():
    st.title("Movie Score Predictor")
    st.write("Predict a movie's score based on its details")

    with st.sidebar:
        st.header("About")
        st.write("This app predicts movie scores using machine learning.")
        st.write("Fill in the details of the movie you want to know the " +
                 "score of and click 'Predict'.")

    # Load model (shows spinner while loading)
    with st.spinner("Loading model..."):
        model = load_model()

    # Input form
    with st.form("movie_input_form"):
        st.subheader("Movie Details")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Movie Title*",
                                 help="Official title of the movie",
                                 value="Inception"
                                 )
            rating = st.selectbox(
                "MPA Rating*",
                ["PG-13", "G", "PG", "R", "NC-17", "Not Rated"],
                index=3,
                help="Motion Picture Association rating"
                )
            genre = st.selectbox(
                "Genre*",
                [
                    "Sci-Fi", "Comedy", "Action", "Drama", "Crime",
                    "Biography", "Adventure", "Animation", "Horror", "Fantasy",
                    "Mystery", "Thriller", "Family", "Romance", "Western",
                    "Musical", "Music", "History", "Sport"
                ],
                index=2,
                help="Main genre of the movie"
            )
            year = st.number_input("Release Year*",
                                   min_value=1970,
                                   max_value=2025,
                                   help="Year of the movie's release",
                                   value=2010)
            runtime = st.number_input("Runtime*", min_value=1,
                                      max_value=300, value=148,
                                      help="Length of the movie in minutes")
            budget = st.number_input("Budget",
                                     min_value=1000,
                                     max_value=1000000000,
                                     value=160000000,
                                     help="Budget of the movie in USD")

        with col2:
            director = st.text_input("Director", "Christopher Nolan")
            writer = st.text_input("Writer", "Christopher Nolan")
            star = st.text_input("Main Star", "Leonardo DiCaprio")
            country = st.text_input("Country", "United States")
            company = st.text_input("Production Company", "Warner Bros.")
            released = st.text_input("Release Date",
                                     help="Date and country of the first \
                                           movie release",
                                     value="July 16, 2010 (United States)")

        # Required fields notice
        st.caption("* Required fields")

        submitted = st.form_submit_button("Predict Score")

    # Handle form submission
    if submitted:
        error = 0
        # Validate required fields
        if not all([name, rating, genre, released, year, runtime]):
            error = 1
            st.error("Please fill in all required fields")
            return

        try:
            # Prepare input dictionary
            inputs = {
                "name": name,
                "rating": rating,
                "genre": genre,
                "year": float(year),
                "released": released,
                "director": director,
                "writer": writer,
                "star": star,
                "country": country,
                "budget": float(budget),
                "company": company,
                "runtime": float(runtime)
            }

            # Make prediction
            with st.spinner("Calculating prediction..."):
                assert model is not None
                prediction = model.predict(inputs)

            # Display results if errors are fixed
            if(error == 0 ):
                st.success(f"Predicted Score: **{prediction:.1f}/10**")
                st.balloons()

        except Exception as e:
            error = 1
            st.error(f"An error occurred: {str(e)}")
if __name__ == "__main__":
    main()
