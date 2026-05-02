import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from src.model import SpamClassifier
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI(title='AI Text Classifier', version='1.0')
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # use ["*"] only for quick local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model on startup
device = torch.device('cpu')
model = SpamClassifier()
model.load_state_dict(torch.load('saved_model/model.pt', map_location=device))
model.eval()
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    explanation: str


def _template_explanation(pred_label: str, confidence: float) -> str:
    if pred_label == "spam":
        return (
            f"The classifier predicts spam with {confidence:.0%} confidence. "
            "The message likely contains patterns commonly seen in promotional "
            "or suspicious content."
        )
    return (
        f"The classifier predicts ham with {confidence:.0%} confidence. "
        "The message appears consistent with normal conversational text."
    )


def _groq_explanation(text: str, pred_label: str, confidence: float) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _template_explanation(pred_label, confidence)

    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    client = Groq(api_key=api_key)

    prompt = (
        "You are helping explain an SMS spam classifier prediction.\n"
        f"Input message: {text}\n"
        f"Predicted label: {pred_label}\n"
        f"Confidence: {confidence:.2%}\n\n"
        "Write a concise explanation in 2-3 sentences.\n"
        "Do not claim certainty. Mention likely language cues and keep it understandable for non-technical users."
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You explain classifier outputs clearly and cautiously."},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content
        if content and content.strip():
            return content.strip()
    except Exception:
        # Keep API resilient if Groq is unavailable or misconfigured.
        pass

    return _template_explanation(pred_label, confidence)

@app.post('/predict', response_model=PredictionResponse)
async def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(400, 'Text cannot be empty')

    # Run classifier
    encoding = tokenizer(
        input.text, return_tensors='pt',
        truncation=True, padding='max_length', max_length=128
    )
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
    probs = torch.softmax(logits, dim=1)[0]
    pred_label = 'spam' if probs[1] > 0.5 else 'ham'
    confidence = probs[1].item() if pred_label == 'spam' else probs[0].item()

    explanation = _groq_explanation(input.text, pred_label, confidence)

    return PredictionResponse(
        label=pred_label,
        confidence=round(confidence, 4),
        explanation=explanation
    )

@app.get('/health')
async def health(): return {'status': 'ok'}
