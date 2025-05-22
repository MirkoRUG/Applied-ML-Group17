import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Load data
movies = pd.read_csv("data.csv")
movies = movies.dropna(subset=["score"])
X = movies.drop("score", axis=1)
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

categories = ["name", "rating", "genre", "released", "director", "writer",
              "star", "country", "company"]

X_train[categories] = X_train[categories].astype(str)
X_test[categories] = X_test[categories].astype(str)


# Train model
model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=categories)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Errorfor the model: {mse}")
dumb_mse = mean_squared_error(y_test, [y_train.mean()] * len(y_test))
print(f"Mean Squared Error for a dumb model: {dumb_mse}")
