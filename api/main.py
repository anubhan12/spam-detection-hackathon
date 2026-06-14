import os
import sys
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Allow imports from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import TextPreprocessor


app = FastAPI(
    title="Spam API",
    description="Production inference engine for filtering incoming spam."
)

model, vectorizer, preprocessor = None, None, None


@app.on_event("startup")
def load_artifacts():
    """Load trained model and vectorizer into memory."""
    global model, vectorizer, preprocessor

    try:
        model = joblib.load("models/model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        preprocessor = TextPreprocessor()
        print("All models successfully loaded into RAM.")

    except FileNotFoundError:
        print("ERROR: Run 'python3 src/train.py' first.")


class SpamRequest(BaseModel):
    text: str


class SpamResponse(BaseModel):
    text: str
    is_spam: bool
    confidence: float


@app.post("/predict", response_model=SpamResponse)
def predict_spam(request: SpamRequest):

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Run training first.")

    # preprocess
    cleaned = preprocessor.clean_text(request.text)

    # vectorize (must be 2D input)
    vectorized_input = vectorizer.transform([cleaned])

    # prediction
    prediction = model.predict(vectorized_input)[0]
    probabilities = model.predict_proba(vectorized_input)[0]

    # correct confidence extraction
    pred_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[pred_index])

    # correct spam check
    is_spam_bool = str(prediction).lower() in ["spam", "1", "true"]

    return SpamResponse(
        text=request.text,
        is_spam=is_spam_bool,
        confidence=round(confidence, 4)
    )