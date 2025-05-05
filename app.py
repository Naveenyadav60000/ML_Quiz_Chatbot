from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model, vectorizer, encoder
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

app = FastAPI()

class Question(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str

@app.post("/predict")
def predict(q: Question):
    combined = f"{q.question} {q.option_a} {q.option_b} {q.option_c} {q.option_d}"
    vec = vectorizer.transform([combined])
    pred = model.predict(vec)
    label = le.inverse_transform(pred)[0]
    return {"predicted_difficulty": label}
