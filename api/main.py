import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference import load_classifier, predict_payload

load_classifier()

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

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    explanation: str

@app.post('/predict', response_model=PredictionResponse)
async def predict(input: TextInput):
    try:
        result = predict_payload(input.text)
    except ValueError:
        raise HTTPException(400, 'Text cannot be empty')

    return PredictionResponse(
        label=result['label'],
        confidence=result['confidence'],
        explanation=result['explanation']
    )

@app.get('/health')
async def health(): return {'status': 'ok'}
