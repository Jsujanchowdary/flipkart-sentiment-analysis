# Sentiment Analysis Model Artifacts

This directory contains the trained machine learning model artifacts for the Sentiment Analysis application. The model is built using XGBoost and TF-IDF vectorization.

## Files

*   **`xgboost_sentiment_model.pkl`**: The trained XGBoost classifier model.
*   **`tfidf_vectorizer.pkl`**: The fitted TF-IDF vectorizer used to transform text data into numerical features.
*   **`label_encoder.pkl`**: The label encoder used to encode/decode the target labels (e.g., converting between "positive"/"negative" and 0/1).

## Requirements

To use these models, you will need the following Python libraries:

*   `joblib`
*   `scikit-learn`
*   `xgboost`
*   `numpy`

## Usage

Here is an example of how to load and use these artifacts for prediction (similar to how it is done in `app.py`):

```python
import joblib
import re

# 1. Load the artifacts
tfidf = joblib.load("model/tfidf_vectorizer.pkl")
label_enc = joblib.load("model/label_encoder.pkl")
xgb_model = joblib.load("model/xgboost_sentiment_model.pkl")

# 2. Define a preprocessing function (should match training)
def preprocess(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 3. Predict
sample_text = "This is a fantastic product!"
processed_text = preprocess(sample_text)

# Transform text using the loaded vectorizer
X = tfidf.transform([processed_text])

# Make prediction
prediction_encoded = xgb_model.predict(X)
prediction_label = label_enc.inverse_transform(prediction_encoded)

print(f"Input: {sample_text}")
print(f"Predicted Label: {prediction_label[0]}")
```

## Integration

These files are currently used by `app.py` to serve predictions via a FastAPI endpoint. Ensure these files are present in the `model/` directory relative to the application entry point.
