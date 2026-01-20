import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="MLOps Housing Price Predictor")

model = joblib.load("exported_model/model.pkl")
preprocessor = joblib.load("exported_model/preprocessor.pkl")

class HouseFeatures(BaseModel):
    MedInc: float = Field(...)
    HouseAge: float = Field(...)
    AveRooms: float = Field(...)
    AveBedrms: float = Field(...)
    Population: float = Field(...)
    AveOccup: float = Field(...)
    Latitude: float = Field(...)
    Longitude: float = Field(...)

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        data = pd.DataFrame([features.dict()])  # preserves column order
        X = preprocessor.transform(data)
        prediction = model.predict(X)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metadata")
def metadata():
    return {
        "model_type": "RandomForestRegressor",
        "features": [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ],
        "mlflow_run_id": "..."
    }
