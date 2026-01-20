import os
import shutil

EXPORT_DIR = "exported_model"
MLFLOW_ARTIFACTS = r"mlruns\0\models\m-dcdb970f38c643f2871e0f3b15c2aea6\artifacts"
LOCAL_MODELS_DIR = "models"

if not os.path.exists(os.path.join(LOCAL_MODELS_DIR, "preprocessor.pkl")):
    raise FileNotFoundError("preprocessor.pkl not found. Run training first.")

if not os.path.exists(os.path.join(LOCAL_MODELS_DIR, "model.pkl")):
    raise FileNotFoundError("model.pkl not found. Run training first.")

# Clean old export
if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

os.makedirs(EXPORT_DIR, exist_ok=True)

# Copy MLflow model files
shutil.copytree(MLFLOW_ARTIFACTS, EXPORT_DIR, dirs_exist_ok=True)

# Copy local artifacts
shutil.copy(
    os.path.join(LOCAL_MODELS_DIR, "preprocessor.pkl"),
    os.path.join(EXPORT_DIR, "preprocessor.pkl")
)

shutil.copy(
    os.path.join(LOCAL_MODELS_DIR, "model.pkl"),
    os.path.join(EXPORT_DIR, "model.pkl")
)

print("MODEL_EXPORTED")
