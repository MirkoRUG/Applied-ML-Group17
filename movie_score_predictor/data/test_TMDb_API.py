import requests

TMDB_API_KEY = "dcbe7f561670aeffbe72a416d3c16a08"
BASE_URL = "https://api.themoviedb.org/3"


def fetch_movie_data(movie_id):
    """Fetches and formats TMDB data for our model."""
    endpoints = [
        f"/movie/{movie_id}",
        f"/movie/{movie_id}/credits"
    ]

    responses = []
    for endpoint in endpoints:
        try:
            response = requests.get(BASE_URL +
                                    endpoint, params={"api_key": TMDB_API_KEY})
            responses.append(response.json())
        except Exception as e:
            print(f"Error fetching {endpoint}: {e}")
            return None

    movie_data, credits_data = responses
    if not (movie_data and credits_data):
        return None

    # Extract minimal features needed for the model
    return {
        'name': movie_data.get('title', 'Unknown'),
        'genre': movie_data.get('genres', [{}])[0].get('name', 'Unknown'),
        'year': int(movie_data.get('release_date', '')[:4])
        if movie_data.get('release_date') else None,
        'director': next((p['name'] for p in credits_data.get('crew', [])
                          if p.get('job') == 'Director'), 'Unknown'),
        'runtime': movie_data.get('runtime')
    }


# Example usage (for testing)
if __name__ == "__main__":
    test_movies = [550, 27205]  # Fight Club, Inception
    for movie_id in test_movies:
        data = fetch_movie_data(movie_id)
        print(f"Movie ID {movie_id}:", data)
