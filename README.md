
# Nirmaan AI – Spoken Introduction Scorer (Hybrid NLP Edition)

This repository contains a **product-style ML solution** for Nirmaan's *spoken self-introduction* case study.

The project is implemented from the perspective of an **ML / NLP engineer**:

- A clean, modular **Python scoring engine** (`scoring.py`)
- A simple **Streamlit web app** (`streamlit_app.py`)
- Hybrid **rule-based + NLP** scoring
- Clear **input → output** contract using JSON
- Ready to run locally, on Colab, or on Render/Railway

---

## 1. Problem statement (short)

Students record a short **self-introduction**. We receive the **transcript**.

We want to automatically score it on:

- Content coverage (name, class, school, hobbies, goals…)
- Structure and clarity
- Speaking style (filler words, approximate length, optional speech rate)
- Semantic quality vs an “ideal” introduction
- Sentiment / engagement (how positive / enthusiastic it sounds)

The output should be:

- A single **score out of 100**
- A **criterion-wise breakdown** with explanations
- Machine-readable **JSON** for downstream systems

---

## 2. High-level design

### Inputs

```python
transcript: str        # required
duration_seconds: float | None = None  # optional (for WPM)
```

### Outputs

The core function is:

```python
from scoring import evaluate_transcript

result = evaluate_transcript(transcript, duration_seconds=55.0)
```

It returns an `EvaluationResult` which can be converted into a dictionary via `.to_dict()`:

```json
{
  "overall_score": 86.3,
  "max_score": 100.0,
  "criteria": [
    {
      "name": "Salutation quality (rule-based)",
      "max_score": 5,
      "score": 4,
      "details": { "tier": "good" },
      "score_normalized_0_1": 0.8
    },
    {
      "name": "Semantic coverage vs ideal introduction (NLP)",
      "max_score": 20,
      "score": 17.1,
      "details": {
        "cosine_similarity": 0.74,
        "normalized_0_1": 0.87
      },
      "score_normalized_0_1": 0.855
    },
    ...
  ],
  "meta": {
    "word_count": 92,
    "duration_seconds": 55.0,
    "raw_total": 77.5,
    "raw_max": 90.0,
    "uses_nlp_models": true
  }
}
```

This is easy to log, feed into dashboards, or expose via an API.

---

## 3. What the scoring engine does

All logic is in `scoring.py`. It has two layers:

### 3.1 Rule-based rubric (transparent)

1. **Salutation quality (0–5)**  
   - “Hi / Hello / Hey” → *normal*  
   - “Good morning / Good afternoon / Good evening” → *good*  
   - More advanced greetings (“I am excited to introduce…”) → *excellent*  

2. **Core content coverage (0–20)**  
   Checks whether the transcript mentions **must-have** slots:
   - Name
   - Class / grade
   - School / college
   - Hobbies / goals  
   Each present slot gets 5 points (max 20).

3. **Additional details (0–8)**  
   Encourages richer introductions:
   - Age
   - Family background
   - Origin city
   - Thanks / closing  
   Each present slot gets 2 points.

4. **Clarity (filler words) (0–10)**  
   Computes filler words per 100 tokens: `um, uh, like, you know, sort of, basically, actually…`  
   Fewer filler words → higher score.

5. **Length appropriateness (0–10)**  
   - Very short (< 30 tokens) → penalised  
   - Ideal range (~40–120 tokens) → full score  
   - Overly long (> 200 tokens) → lightly penalised

6. **Vocabulary richness – TTR (0–10)**  
   Type–Token Ratio (unique_words / total_words) → mapped to buckets (0–2–4–6–8–10).

7. **Speech rate (optional, 0–10)**  
   If `duration_seconds` is provided, we estimate **Words Per Minute** and map to a score.  
   If not provided, this criterion is marked as `"not_applicable"` and excluded from the max.

### 3.2 NLP layer (semantic + sentiment)

1. **Semantic coverage vs ideal introduction (0–20)**  
   - Uses `sentence-transformers/all-MiniLM-L6-v2` to compute a sentence embedding for:
     - Your transcript
     - A short description of an “ideal student introduction”  
   - Cosine similarity is then mapped to a score in 0–20.  
   - This captures *how close* the content feels to a high-quality introduction in meaning, not just keywords.

2. **Engagement / positivity (sentiment) (0–20)**  
   - Uses Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` via `transformers.pipeline("sentiment-analysis")`.  
   - We interpret the probability of the positive class as a **positivity index**, and map it to 0–20.  
   - This acts as a proxy for **enthusiasm, confidence, and friendly tone**.

### 3.3 Overall score

- Sum all available criterion scores.
- Divide by sum of their max scores (ignoring metrics marked `not_applicable`).
- Scale to 0–100.

This gives a stable, interpretable score where:

- Rule-based features keep alignment with the rubric.
- NLP features reward good semantic content and tone.

---

## 4. Running the project locally

### 4.1. Environment (conda example)

```bash
conda create -n nirmaan_intro_nlp python=3.10
conda activate nirmaan_intro_nlp
```

### 4.2. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will download Hugging Face models (`sentence-transformers` and `transformers`).

### 4.3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Then open the URL shown in your terminal (usually http://localhost:8501).

### 4.4. Using the app

1. Paste or upload a student’s transcript (`.txt`).
2. (Optional) Enter the audio duration in seconds.
3. Click **“Score introduction ✅”**.
4. View:
   - Overall score
   - Criterion-wise table
   - Detailed JSON-style explanation under each expander
   - Raw JSON (for debugging or API usage)

---

## 5. Deploying to Render / Railway

A typical configuration:

**Build command**

```bash
pip install -r requirements.txt
```

**Start command**

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

Steps:

1. Push this project to **GitHub**.
2. On Render / Railway, create a **Web Service** from that repo.
3. Set the build and start commands above.
4. Deploy and grab the public URL for your case-study submission.

---

## 6. Repository structure

```text
nirmaan_intro_nlp_project/
├── scoring.py                        # Hybrid rule-based + NLP scoring engine
├── streamlit_app.py                  # Streamlit interface
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── DEPLOYMENT.md                     # Short deployment guide
├── LICENSE                           # MIT license
├── (optional) Nirmaan_Intro_Scorer_Colab.ipynb   # Colab demo notebook
└── rubric & sample files (outside this folder, but in repo root)
    ├── Case study for interns.xlsx
    ├── Sample text for case study.txt
    └── Nirmaan AI intern Case study instructions.pdf
```

You can keep the rubric & PDF in the repo root or under a `data/` folder and adjust paths
accordingly if you later load rubric thresholds programmatically.

---

## 7. Extensibility notes (for reviewers)

If we wanted to take this prototype further, we could:

- Replace heuristics with **learned models** (e.g., train a small regression model per criterion).
- Add a **grammar / fluency** model (e.g. GEC or language-tool) to score correctness.
- Use **ASR confidences + pause detection** to incorporate actual audio properties.
- Log large volumes of scored transcripts and perform **rubric calibration** with educators.
- Wrap this module in a small **FastAPI** service for scalable deployment.

For the internship case study, the goal is to show **end-to-end product thinking**:
clean interfaces, explainable scoring, and realistic use of NLP models.
