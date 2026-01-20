# MLOps End-to-End Housing Price Prediction Pipeline

An end-to-end MLOps project demonstrating data preprocessing, model training, experiment tracking with MLflow, model export, and a production-ready FastAPI service deployed via Docker.

---

## 🔹 Project Overview

This project covers the full lifecycle of a machine learning system:

* Data ingestion and preprocessing
* Model training using Scikit-learn
* Experiment tracking using MLflow
* Model export for inference
* REST API using FastAPI
* Containerization using Docker

The system predicts California housing prices based on input features.

---

## 🔹 Repository Visibility

✅ **This GitHub repository is PUBLIC.**

Anyone can:

* View the source code
* Clone the project
* Fork and modify it
* Run it locally

Repo URL:

```
https://github.com/FD785/mlops-end-to-end-pipeline
```

---

## 🔹 API Accessibility (Important)

⚠️ **The FastAPI service is NOT public yet.**

Right now the API runs locally at:

```
http://127.0.0.1:8000/docs
```

That means:

| Component       | Public? | Reason                          |
| --------------- | ------- | ------------------------------- |
| GitHub repo     | ✅ Yes   | Repository visibility is Public |
| FastAPI service | ❌ No    | Bound to localhost (127.0.0.1)  |
| Swagger UI      | ❌ No    | Only visible on your machine    |
| Model endpoint  | ❌ No    | Not deployed to the cloud       |

The API is only accessible on the machine where Docker is running.

---

## 🔹 How Others Can Use This Project

### Option 1 — Run Locally

```bash
# Clone repo
git clone https://github.com/FD785/mlops-end-to-end-pipeline.git
cd mlops-end-to-end-pipeline

# Build Docker image
docker build -t housing-api .

# Run API container
docker run -p 8000:8000 housing-api
```

Then open:

```
http://127.0.0.1:8000/docs
```

---

### Option 2 — Use the API Endpoint

POST request:

```
POST http://127.0.0.1:8000/predict
```

Example JSON payload:

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.984127,
  "AveBedrms": 1.02381,
  "Population": 322,
  "AveOccup": 2.555556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

---

## 🔹 Cloud Deployment (Future Scope)

To make the API publicly accessible, it can be deployed to:

* AWS / GCP / Azure
* Railway / Render
* Fly.io
* EC2 + Docker
* Kubernetes

Once deployed, the API would be reachable via a public URL such as:

```
https://housing-api.yourdomain.com/predict
```

---

## 🔹 Tech Stack

* Python
* Scikit-learn
* MLflow
* FastAPI
* Docker
* Pandas
* NumPy

---


---

## 🔹 License

MIT License

