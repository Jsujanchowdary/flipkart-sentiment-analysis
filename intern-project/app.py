from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import joblib
import numpy as np

# Load model artifacts
TFIDF_PATH = "model/tfidf_vectorizer.pkl"
LABEL_PATH = "model/label_encoder.pkl"
MODEL_PATH = "model/xgboost_sentiment_model.pkl"

tfidf = joblib.load(TFIDF_PATH)
label_enc = joblib.load(LABEL_PATH)
xgb_model = joblib.load(MODEL_PATH)

app = FastAPI(title="Sentiment Model API")

class SingleIn(BaseModel):
    text: str

class BatchIn(BaseModel):
    texts: List[str]

@app.post("/predict")
def predict(payload: Union[SingleIn, BatchIn]):
    """
    Accepts either {"text": "..."} or {"texts": ["a","b",...]}
    Returns {"prediction": 1} or {"predictions": [1,0,...]} (encoded ints)
    and {"labels": ["positive", ...]} for decoded labels
    """
    try:
        if hasattr(payload, "text") and payload.text is not None:
            texts = [payload.text]
        elif hasattr(payload, "texts") and payload.texts is not None:
            texts = payload.texts
        else:
            raise HTTPException(status_code=400, detail="Missing 'text' or 'texts'")

        # Preprocessing: same as training. Simple cleaning example:
        import re
        def preprocess(t):
            t = re.sub('[^a-zA-Z0-9]', ' ', t)
            t = re.sub(r'\s+', ' ', t)
            return t.strip()

        texts_proc = [preprocess(t) for t in texts]
        X = tfidf.transform(texts_proc)
        preds_encoded = xgb_model.predict(X)
        preds_labels = label_enc.inverse_transform(preds_encoded)

        # Return both encoded and labels
        if len(preds_encoded) == 1:
            return {"prediction": int(preds_encoded[0]), "label": str(preds_labels[0])}
        else:
            return {
                "predictions": [int(intp) for intp in preds_encoded],
                "labels": [str(l) for l in preds_labels]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
