import sys
from pathlib import Path

import gradio as gr

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference import load_classifier, predict_payload

load_classifier()


def classify_text(text):
    if not text.strip():
        return 'Please enter some text', '', ''
    try:
        data = predict_payload(text)
    except Exception:
        return 'Error', '', 'Classification failed.'
    label = data['label'].upper()
    conf = f"{data['confidence']:.1%}"
    explanation = data['explanation']
    return label, conf, explanation

demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(placeholder='Paste any SMS message here...', lines=3),
    outputs=[
        gr.Textbox(label='Prediction'),
        gr.Textbox(label='Confidence'),
        gr.Textbox(label='AI Explanation', lines=4),
    ],
    title='SMS Spam Classifier',
    description='Classifies SMS messages as spam or ham, with an AI-generated explanation.',
    examples=[
        ['Congratulations! You won a free iPhone. Click here to claim.'],
        ['Hey, are we still on for dinner tonight?'],
        ['URGENT: Your bank account has been suspended. Verify now.'],
    ]
)

if __name__ == '__main__':
    demo.launch()
