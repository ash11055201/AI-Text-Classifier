"""Shared spam classification + explanations for API and Gradio."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from groq import Groq
from transformers import AutoTokenizer

from .model import SpamClassifier

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEIGHTS_PATH = _REPO_ROOT / "saved_model" / "model.pt"

_device = torch.device("cpu")
_model: SpamClassifier | None = None
_tokenizer = None


def load_classifier() -> None:
    """Load weights and tokenizer once (call at app startup)."""
    global _model, _tokenizer
    if _model is not None:
        return
    _model = SpamClassifier()
    _model.load_state_dict(torch.load(_WEIGHTS_PATH, map_location=_device))
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


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
        pass

    return _template_explanation(pred_label, confidence)


def predict_payload(text: str) -> dict:
    """
    Run the classifier on non-empty text.
    Returns dict: label ('spam'|'ham'), confidence (float), explanation (str).
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    load_classifier()
    assert _model is not None and _tokenizer is not None

    encoding = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    with torch.no_grad():
        logits = _model(encoding["input_ids"], encoding["attention_mask"])
    probs = torch.softmax(logits, dim=1)[0]
    pred_label = "spam" if probs[1] > 0.5 else "ham"
    confidence = probs[1].item() if pred_label == "spam" else probs[0].item()
    explanation = _groq_explanation(text, pred_label, confidence)
    return {
        "label": pred_label,
        "confidence": round(confidence, 4),
        "explanation": explanation,
    }
