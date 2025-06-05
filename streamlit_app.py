import streamlit as st
from notebooks.temp_model import get_model, make_prediction

# App configuration
st.set_page_config(page_title="Movie Score Predictor", page_icon="🎬")


# Load your model (modify this function with your actual loading code)
@st.cache_resource  # Cache the model load
def load_model():
    try:
        model = get_model()  # Call your actual model loading function
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None


def main():
    st.title("🎬 Movie Score Predictor")
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
            released = st.text_input("Release Year*",
                                     help="Year of the movie's release",
                                     value="2010")
            runtime = st.number_input("Runtime*", min_value=1,
                                      max_value=300, value=148,
                                      help="Length of the movie in minutes")

        with col2:
            director = st.text_input("Director", "Christopher Nolan")
            writer = st.text_input("Writer", "Christopher Nolan")
            star = st.text_input("Main Star", "Leonardo DiCaprio")
            country = st.selectbox("Country", "United States")
            company = st.text_input("Production Company", "Warner Bros.")

        # Required fields notice
        st.caption("* Required fields")

        submitted = st.form_submit_button("Predict Score")

    # Handle form submission
    if submitted:
        # Validate required fields
        if not all([name, rating, genre, released, runtime]):
            st.error("Please fill in all required fields")
            return

        try:
            # Prepare input dictionary
            inputs = {
                "name": name,
                "rating": rating,
                "genre": genre,
                "released": released,
                "director": director,
                "writer": writer,
                "star": star,
                "country": country,
                "company": company,
                "runtime": float(runtime)
            }

            # Make prediction
            with st.spinner("Calculating prediction..."):
                prediction = make_prediction(model, inputs)

            # Display results
            st.success(f"Predicted Score: **{prediction:.1f}/10**")
            st.balloons()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
