# AI Text Classifier

An **SMS spam vs. ham** classifier backed by a fine-tuned **DistilBERT** model, with a shared inference layer used by a **FastAPI** service and **Gradio** UIs. Optional **Groq**-powered natural-language explanations fall back to a simple template when no key is set.

## What it does

1. **Core model** — Text is tokenized with `distilbert-base-uncased` and passed through a custom classifier (`SpamClassifier`: DistilBERT encoder + dropout + linear head). The API returns **spam** or **ham**, a **confidence** score, and a short **explanation**.
2. **Explanations** — If `GROQ_API_KEY` is set, a Groq chat model (default `llama-3.1-8b-instant`, overridable with `GROQ_MODEL`) writes a cautious 2–3 sentence explanation for non-technical readers. If the key is missing or the call fails, the API falls back to a simple template based on label and confidence.
3. **Training** — `src/train.py` loads the [SMS Spam Collection](https://huggingface.co/datasets/sms_spam) from Hugging Face, fine-tunes the model for a few epochs, and saves weights for inference.

## Project layout

| Path | Role |
|------|------|
| `src/model.py` | `SpamClassifier` PyTorch module |
| `src/inference.py` | Load weights, run classifier, template/Groq explanations |
| `src/train.py` | Training script (HF dataset → `model.pt` in the working directory) |
| `app.py` | **Gradio entrypoint** (single process; use on **Hugging Face Spaces**) |
| `api/main.py` | FastAPI app: `/predict`, `/health` |
| `api/app.py` | Local Gradio UI (same in-process inference as `app.py`) |
| `notebooks/` | EDA and evaluation notebooks |
| `saved_model/model.pt` | Trained weights (not committed by default; see below) |

## Prerequisites

- Python 3.10+ recommended  
- A virtual environment (`venv/` is gitignored)  
- Trained weights at **`saved_model/model.pt`** (see [Model weights](#model-weights))

Runtime (API + Gradio + inference): see `requirements.txt`. Training additionally needs `datasets` and `scikit-learn`.

## Setup

1. Clone the repo and create a venv:

   ```bash
   python -m venv venv
   ```

   On Windows: `venv\Scripts\activate`  
   On macOS/Linux: `source venv/bin/activate`

2. Install dependencies (adjust versions to match your CUDA/CPU PyTorch build if needed):

   ```bash
   pip install -r requirements.txt
   ```

   For training:

   ```bash
   pip install datasets scikit-learn
   ```

3. **Model weights** — After training (see below), ensure the checkpoint is available as:

   `saved_model/model.pt`

   Training currently writes `model.pt` in the directory you run from (e.g. `src/`); copy or move it into `saved_model/` so the API path matches.

4. **Optional: Groq explanations** — Set environment variables (e.g. in a `.env` file if you load it with your shell or a tool like `python-dotenv`; `.env` is gitignored):

   - `GROQ_API_KEY` — required for LLM explanations  
   - `GROQ_MODEL` — optional; defaults to `llama-3.1-8b-instant`

## Run the API

From the **repository root** (so `src` imports resolve):

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

- **POST** `/predict` — JSON body: `{"text": "your SMS or message here"}`  
  Response: `label` (`spam` | `ham`), `confidence`, `explanation`  
- **GET** `/health` — `{"status": "ok"}`

CORS is enabled for local frontends on ports **3000** and **5173**.

## Run the Gradio demo

Gradio calls the model **in-process** (no FastAPI required).

From the **repository root**:

```bash
python app.py
```

Uses `PORT` if set (e.g. on Hugging Face Spaces); otherwise opens on port **7860** with `server_name="0.0.0.0"` so the Space can reach the app.

Alternatively:

```bash
python api/app.py
```

Open the URL Gradio prints, paste text, and view prediction, confidence, and explanation.

## Hugging Face Spaces

1. **SDK** — Create a **Gradio** Space; set the app file to **`app.py`** (default when `app.py` exists at the repo root).
2. **Dependencies** — Spaces installs from **`requirements.txt`** at the repo root. Keep it in sync with imports in `app.py` / `src/inference.py`.
3. **Weights** — The app loads **`saved_model/model.pt`** relative to the repo root. Your local `.gitignore` ignores `saved_model/`; for Spaces you still need that file **in the Space repo**, for example:
   - [Git LFS](https://git-lfs.com/) for `saved_model/model.pt`, or  
   - `git add -f saved_model/model.pt` on a branch used only for deployment, or  
   - a build step that downloads the checkpoint from private storage.
4. **Secrets** — In the Space **Settings → Secrets**, add **`GROQ_API_KEY`** (and optionally **`GROQ_MODEL`**) if you want LLM explanations; otherwise template explanations are used.
5. **One process** — Do not rely on a separate `uvicorn` process; the root **`app.py`** path is the supported Space layout.

## Train the model

```bash
cd src
python train.py
```

This downloads `sms_spam`, trains for 3 epochs, and prints validation accuracy/F1 each epoch. Place the saved `model.pt` at **`saved_model/model.pt`** before starting the API.

## Notebooks

Use `notebooks/text_classifier_eda.ipynb` and `notebooks/text_classifier_evaluation.ipynb` for exploratory analysis and evaluation.

## Git and large artifacts

`saved_model/` and `venv/` are listed in `.gitignore` so large binaries and environments are not pushed by default. For **Hugging Face Spaces**, you still need a strategy to supply `saved_model/model.pt` in the deployed repo (see [Hugging Face Spaces](#hugging-face-spaces)).
