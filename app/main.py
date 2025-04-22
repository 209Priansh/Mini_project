from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_sentiment

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "TinyBERT Sentiment Analysis API is running."}

@app.post("/predict/")
def predict(input: TextInput):
    prediction = predict_sentiment(input.text)
    return {"sentiment": prediction}
