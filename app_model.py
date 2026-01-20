import os
import joblib
import json

MODEL_PATH = os.path.join("models", "model.joblib")
FEATURES_PATH = os.path.join("models", "feature_names.json")

def load_artifact():
    """
    Load model artifact if present. Returns dict:
    {
      "pipeline": sklearn Pipeline,
      "feature_names": [...],
      "meta": {...}
    }
    If not found, return a dummy predictor artifact.
    """
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data
    else:
        # Dummy artifact: predict constant 0.0 and use default Cali feature names
        default_features = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
        class DummyPipeline:
            def predict(self, X):
                # X: array-like
                return [0.0 for _ in range(len(X))]
        return {
            "pipeline": DummyPipeline(),
            "feature_names": default_features,
            "meta": {"name": "dummy", "version": "0.0"}
        }

_artifact = load_artifact()

def get_feature_names():
    return _artifact.get("feature_names", [])

def predict_from_dict(feature_dict):
    """
    Accepts dict of feature_name -> value, orders features, returns single prediction (float)
    """
    feature_names = get_feature_names()
    # Order values according to feature_names
    X = []
    for fn in feature_names:
        v = feature_dict.get(fn)
        if v is None:
            raise ValueError(f"Missing feature: {fn}")
        X.append(float(v))
    # Prepare 2D array for sklearn
    X2 = [X]
    pipeline = _artifact["pipeline"]
    preds = pipeline.predict(X2)
    return float(preds[0])

def get_model_meta():
    return _artifact.get("meta", {"name": "unknown", "version": "0.0"})