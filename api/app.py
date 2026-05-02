
import gradio as gr
import requests

API_URL = 'http://localhost:8000/predict'

def classify_text(text):
    if not text.strip():
        return 'Please enter some text', 0.0, ''
    response = requests.post(API_URL, json={'text': text})
    if response.status_code != 200:
        return 'Error calling API', 0.0, ''
    data = response.json()
    label = data['label'].upper()
    conf  = f"{data['confidence']:.1%}"
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

demo.launch()
