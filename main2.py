from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app import model as model_module
from typing import Dict
import uvicorn

app = FastAPI(title="UAS ML Housing - Prediction API")

@app.get("/health")
def health():
    meta = model_module.get_model_meta()
    return {"status": "ok", "model": meta.get("name"), "version": meta.get("version")}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Validate keys and types minimally
    try:
        pred = model_module.predict_from_dict(req.features)
        meta = model_module.get_model_meta()
        return PredictResponse(prediction=pred, model=meta.get("name"), version=meta.get("version"), inputs=req.features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction error: " + str(e))

# If run directly: start uvicorn (useful for local dev)
if _name_ == "_main_":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)