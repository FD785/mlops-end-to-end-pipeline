import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
    data_path = os.path.join(base_dir, "data", "housing.csv")

    df = pd.read_csv(data_path)

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    numeric_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)]
    )

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor
