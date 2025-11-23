
"""
Streamlit UI for Nirmaan AI – Spoken Intro Scorer (NLP Edition)
"""

import streamlit as st
from io import StringIO
from scoring import evaluate_transcript

st.set_page_config(page_title="Nirmaan AI – Intro Scorer (NLP)", layout="wide")

st.title("Nirmaan AI – Spoken Introduction Scorer")
st.caption("Hybrid rule-based + NLP scoring engine for student self-introductions.")

with st.sidebar:
    st.header("Options")
    duration = st.number_input(
        "Audio duration (seconds, optional)",
        min_value=0.0,
        value=0.0,
        help="Used to estimate words-per-minute. Leave 0 if unknown.",
    )
    show_raw = st.checkbox("Show raw JSON output", value=True)

tab_input, tab_results = st.tabs(["1️⃣ Input transcript", "2️⃣ Results & explanation"])

with tab_input:
    st.subheader("Transcript input")

    uploaded = st.file_uploader("Upload transcript (.txt)", type=["txt"])
    default_text = ""
    if uploaded is not None:
        default_text = StringIO(uploaded.getvalue().decode("utf-8", errors="ignore")).read()

    transcript = st.text_area(
        "Paste transcript or edit uploaded text",
        value=default_text,
        height=260,
        placeholder="Example: Good morning, my name is ...",
    )

    run = st.button("Score introduction ✅")

with tab_results:
    if run:
        if not transcript.strip():
            st.error("Please paste or upload a transcript first.")
        else:
            with st.spinner("Loading NLP models and scoring (first run may take a bit)..."):
                result = evaluate_transcript(transcript, duration_seconds=(duration or None))
            st.session_state["last_result"] = result.to_dict()

    if "last_result" not in st.session_state:
        st.info("Run the scorer from the **Input transcript** tab to see results here.")
    else:
        res = st.session_state["last_result"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall score (0–100)", f"{res['overall_score']:.1f}")
        with col2:
            st.write("Meta")
            st.json(res["meta"])

        st.markdown("---")
        st.subheader("Rubric & NLP breakdown")

        rows = []
        for c in res["criteria"]:
            rows.append(
                {
                    "Criterion": c["name"],
                    "Score": c["score"],
                    "Max": c["max_score"],
                    "Normalised (0–1)": c["score_normalized_0_1"],
                }
            )
        st.dataframe(rows, use_container_width=True)

        st.subheader("Detailed explanation")
        for c in res["criteria"]:
            with st.expander(f"{c['name']} – {c['score']}/{c['max_score']}"):
                st.json(c["details"])

        if show_raw:
            st.subheader("Raw JSON output")
            st.json(res)
