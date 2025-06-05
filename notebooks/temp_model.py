from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import pandas as pd


def get_model():
    # ### !!!!! ###### THIS IS NOT CORRECT, @WILL FIX YOUR SHIT
    # (ask Mirko if you're lost)

    # Load data
    movies = pd.read_csv("data.csv")
    movies = movies.dropna(subset=["score"])
    X = movies.drop(["score", "votes", "gross", "released", "budget"], axis=1)
    y = movies["score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    categories = ["name", "rating", "genre", "director", "writer",
                  "star", "country", "company"]
    X_train[categories] = X_train[categories].astype(str)
    X_test[categories] = X_test[categories].astype(str)

    # Initialize model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',  # or MAE, Quantile, etc.
        verbose=100
    )
    # Train model
    model.fit(X_train, y_train, eval_set=(X_test, y_test),
              cat_features=categories)

    return model


# Prediction function (modify with your actual prediction code)
def make_prediction(model, inputs: dict) -> float:
    return model.predict(pd.Series(inputs))
    # Your prediction logic here
    # return model.predict([[...]])[0]
    return 7.5  # Placeholder


# ============================================================================#
# ============================================================================#
# ============================================================================#
# ============================================================================#


def main():
    from sklearn.metrics import mean_squared_error

    # Load data
    movies = pd.read_csv("data.csv")
    movies = movies.dropna(subset="score")
    X = movies.drop(["score", "votes", "gross", "released", "budget"], axis=1)
    y = movies["score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    # Initialize model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',  # or MAE, Quantile, etc.
        verbose=100
    )

    categories = ["name", "rating", "genre", "director", "writer",
                  "star", "country", "company"]

    X_train[categories] = X_train[categories].astype(str)
    X_test[categories] = X_test[categories].astype(str)

    # Train model
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2,
                                                      random_state=42)
    model.fit(X_train, y_train, eval_set=(X_val, y_val),
              cat_features=categories)

    # Predict
    y_pred = model.predict(X_test)

    print(X_test.columns)
    print(X_test.head())

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Errorfor the model: {mse}")
    dumb_mse = mean_squared_error(y_test, [y_train.mean()] * len(y_test))
    print(f"Mean Squared Error for a dumb model: {dumb_mse}")


if __name__ == "__main__":
    main()
