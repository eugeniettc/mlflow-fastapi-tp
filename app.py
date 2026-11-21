from typing import List
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Digits Classifier API")

model = mlflow.sklearn.load_model("model")


class PredictRequest(BaseModel):
    features: List[float]


@app.post("/predict")
def predict(request: PredictRequest):
    if len(request.features) != 64:
        raise HTTPException(
            status_code=400,
            detail=f"64 features attendues (dataset digits), reçu {len(request.features)}.",
        )
    data = np.array(request.features, dtype=float).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}