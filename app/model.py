from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def predict_sentiment(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs).item()
    
    # Class range: 1-5 stars → Map to sentiment
    if predicted_class <= 1:
        return "negative"
    elif predicted_class == 2:
        return "neutral"
    else:
        return "positive"
