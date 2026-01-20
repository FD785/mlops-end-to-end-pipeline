import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

from preprocess import load_and_preprocess_data


def train_model():
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        os.makedirs("models", exist_ok=True)

        # 🔴 THESE TWO LINES ARE THE KEY FIX
        joblib.dump(model, "models/model.pkl")
        joblib.dump(preprocessor, "models/preprocessor.pkl")

        mlflow.sklearn.log_model(model, "model")

        print("Training complete.")
        print(f"MSE: {mse}")
        print(f"R2: {r2}")


if __name__ == "__main__":
    train_model()
