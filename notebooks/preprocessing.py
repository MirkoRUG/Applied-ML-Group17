from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, \
                                  StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from CustomTransformer import TopCategoryEncoder
import pandas as pd


movies = pd.read_csv('data.csv')
df = movies.dropna(subset=["score"])
X = df.drop(["name", "score", "released", "votes", "gross", "country"], axis=1)
y = df["score"]


def convert_ratings(X: pd.DataFrame) -> pd.DataFrame:
    rating_map = {
        "G": "G",
        "PG": "PG",
        "PG-13": "PG-13",
        "R": "R",
        "NC-17": "NC-17",
        "X": "R",               # treat like R
        "Approved": "PG",       # treat like PG
        "TV-PG": "PG",          # treat like PG
        "TV-14": "PG-13",       # treat like PG-13
        "TV-MA": "R",           # treat like R
        "Not Rated": "Unrated",
        "Unrated": "Unrated"
    }
    converted_data = X["rating"].map(rating_map).fillna("Unrated")
    return pd.DataFrame({"rating": converted_data})


X_train, X_test, y_train, y_test = train_test_split(X, y)

rating_pipeline = Pipeline([
    ("convert", FunctionTransformer(convert_ratings)),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

column_trans = ColumnTransformer([
    ("convert ratings", rating_pipeline, ["rating"]),
    ("director_top_cat", TopCategoryEncoder(), ["director"]),
    ("writer_top_cat", TopCategoryEncoder(), ["writer"]),
    ("star_top_cat", TopCategoryEncoder(), ["star"]),
    ("company_top_cat", TopCategoryEncoder(), ["company"]),
    ("one hot encoder", OneHotEncoder(handle_unknown='ignore'), ["genre"]),
    ("standard scaler", StandardScaler(), ["year", "runtime", "budget"])]
    # , remainder='passthrough'
)

pipe = Pipeline([
    ("column transformer", column_trans),
    ("random forest regressor", RandomForestRegressor())
])


print(y.isna().sum())

pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
