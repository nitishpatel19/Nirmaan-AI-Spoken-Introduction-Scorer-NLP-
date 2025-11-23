
# Deployment Guide – Nirmaan AI Spoken Intro Scorer (NLP Edition)

## 1. Local development

```bash
conda create -n nirmaan_intro_nlp python=3.10
conda activate nirmaan_intro_nlp
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the URL shown in the terminal (e.g. http://localhost:8501).

## 2. Render / Railway (recommended for demo URL)

**Build command**

```bash
pip install -r requirements.txt
```

**Start command**

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

Set Python version to 3.10+ in your service settings. First deploy will also download
Hugging Face models (this may take a couple of minutes).

## 3. Colab (optional)

1. Upload this project as a GitHub repo.
2. In Colab, open the repo via “File → Open notebook → GitHub”.
3. You can:
   - Import `evaluate_transcript` directly, or
   - Run a lightweight Colab demo notebook.

For the official submission, prefer sharing the **GitHub repo** + **hosted Streamlit URL**.
