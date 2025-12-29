import os
import re
import pickle
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
import pdfplumber
import docx


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# Premium CSS
# =========================
st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(1200px circle at 10% 10%, rgba(102,126,234,0.35), transparent 60%),
                    radial-gradient(900px circle at 90% 30%, rgba(118,75,162,0.35), transparent 55%),
                    linear-gradient(135deg, #0b1020 0%, #11162a 45%, #0b1020 100%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .wrap {
        max-width: 1100px;
        margin: 0 auto;
        padding: 1.25rem 0 0.25rem 0;
    }

    .hero {
        background: linear-gradient(135deg, rgba(102,126,234,0.22), rgba(118,75,162,0.18));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 22px;
        padding: 1.6rem 1.6rem;
        box-shadow: 0 18px 65px rgba(0,0,0,0.35);
        backdrop-filter: blur(8px);
    }

    .title {
        font-size: 2.6rem;
        font-weight: 850;
        color: rgba(255,255,255,0.96);
        letter-spacing: 0.2px;
        margin: 0;
        line-height: 1.1;
    }

    .subtitle {
        margin: 0.55rem 0 0 0;
        color: rgba(255,255,255,0.78);
        font-size: 1.02rem;
        font-weight: 350;
    }

    .card {
        border-radius: 22px;
        padding: 1.35rem 1.35rem;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 18px 65px rgba(0,0,0,0.30);
        backdrop-filter: blur(8px);
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.14);
        color: rgba(255,255,255,0.86);
        font-size: 0.85rem;
        margin-right: 0.4rem;
        margin-top: 0.35rem;
    }

    .resultBox {
        background: linear-gradient(135deg, rgba(102,126,234,0.95), rgba(118,75,162,0.95));
        border-radius: 18px;
        padding: 1.25rem 1.2rem;
        box-shadow: 0 14px 45px rgba(102,126,234,0.22);
        border: 1px solid rgba(255,255,255,0.18);
    }
    .resultLabel {
        color: rgba(255,255,255,0.9);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.3rem;
        font-weight: 650;
    }
    .resultValue {
        color: white;
        font-size: 2.0rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.1;
    }

    .muted {
        color: rgba(255,255,255,0.70);
        font-size: 0.92rem;
    }

    /* Better buttons */
    .stButton > button {
        width: 100%;
        border: 0;
        border-radius: 12px;
        padding: 0.75rem 1.1rem;
        font-weight: 750;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 12px 30px rgba(102,126,234,0.28);
        transition: transform 0.16s ease, box-shadow 0.16s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 40px rgba(102,126,234,0.40);
    }

    /* Make upload widget look cleaner */
    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 18px !important;
        border: 1.8px dashed rgba(255,255,255,0.30) !important;
        background: rgba(255,255,255,0.06) !important;
        padding: 1.0rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Load saved artifacts
# =========================
@st.cache_resource
def load_artifacts():
    base = Path("models")
    model_path = base / "best_model.pkl"
    vec_path = base / "tfidf_vectorizer.pkl"
    le_path = base / "label_encoder.pkl"

    missing = [p for p in [model_path, vec_path, le_path] if not p.exists()]
    if missing:
        msg = "\n".join([f"- {p.as_posix()}" for p in missing])
        st.error(
            "❌ Model files not found.\n\n"
            "Make sure you have these files inside a folder named **models/**:\n"
            f"{msg}\n\n"
            "Example:\n"
            "models/best_model.pkl\n"
            "models/tfidf_vectorizer.pkl\n"
            "models/label_encoder.pkl"
        )
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        tfidf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    return model, tfidf, le


model, tfidf, le = load_artifacts()

# =========================
# Helpers
# =========================
def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()

    # Normalize whitespace
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Keep useful symbols for tech resumes
    text = re.sub(r"[^a-z0-9\s\.,\-\+\#\/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_pdf(file_bytes: bytes) -> str:
    parts = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
    except Exception as e:
        st.error(f"PDF extract error: {e}")
        return ""
    return "\n".join(parts).strip()


def extract_text_docx(file_bytes: bytes) -> str:
    try:
        d = docx.Document(BytesIO(file_bytes))
        parts = [p.text for p in d.paragraphs if p.text]
        return "\n".join(parts).strip()
    except Exception as e:
        st.error(f"DOCX extract error: {e}")
        return ""


def extract_text_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        st.error(f"TXT extract error: {e}")
        return ""


def extract_text_doc_using_textract(file_bytes: bytes, suffix: str = ".doc") -> str:
    # textract expects a real file path most of the time
    try:
        import textract  # optional dependency
    except Exception:
        return ""  # silently fail -> handled by caller

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        raw = textract.process(tmp_path)
        return raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def extract_text_router(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_pdf(file_bytes)
    if ext == ".docx":
        return extract_text_docx(file_bytes)
    if ext == ".txt":
        return extract_text_txt(file_bytes)
    if ext == ".doc":
        # best-effort via textract
        return extract_text_doc_using_textract(file_bytes, suffix=".doc")
    return ""


def predict_label(cleaned_text: str):
    vec = tfidf.transform([cleaned_text])
    pred_id = model.predict(vec)[0]
    pred_label = le.inverse_transform([pred_id])[0]

    # Confidence / top-3 if predict_proba exists
    top = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(vec)[0]  # shape [n_classes]
            pairs = list(enumerate(proba))
            pairs.sort(key=lambda x: x[1], reverse=True)

            top = []
            for class_idx, p in pairs[:3]:
                label = le.inverse_transform([class_idx])[0]
                top.append((label, float(p)))

        except Exception:
            top = None

    return pred_label, top


# =========================
# Session State
# =========================
defaults = {
    "file_name": None,
    "file_bytes": None,
    "extracted_text": "",
    "cleaned_text": "",
    "prediction": None,
    "top3": None,
    "processed": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_all():
    for k, v in defaults.items():
        st.session_state[k] = v


# =========================
# UI
# =========================
st.markdown(
    """
<div class="wrap">
  <div class="hero">
    <div class="title">📄 AI Resume Classifier</div>
    <div class="subtitle">Upload a resume → extract text → predict job category (TF-IDF + trained ML model)</div>
    <div style="margin-top: 0.6rem;">
      <span class="chip">⚡ Fast</span>
      <span class="chip">🔒 Local Processing</span>
      <span class="chip">🎯 Top-3 Confidence</span>
      <span class="chip">🧾 Export Report</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")
left, mid, right = st.columns([1, 2.2, 1])

with mid:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Upload
    uploaded_file = st.file_uploader(
        "Upload resume",
        type=["pdf", "docx", "doc", "txt"],
        label_visibility="collapsed",
        help="Supported: PDF, DOCX, DOC (best-effort), TXT",
    )

    # Read bytes ONCE (fixes the common Streamlit bug)
    if uploaded_file is not None:
        if (
            st.session_state.file_name != uploaded_file.name
            or st.session_state.file_bytes is None
        ):
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_bytes = uploaded_file.getvalue()
            # reset old results when a new file is selected
            st.session_state.processed = False
            st.session_state.prediction = None
            st.session_state.top3 = None
            st.session_state.extracted_text = ""
            st.session_state.cleaned_text = ""

        info1, info2 = st.columns([3, 1])
        with info1:
            kb = len(st.session_state.file_bytes) / 1024
            st.info(f"📎 **{st.session_state.file_name}**  •  **{kb:.2f} KB**")
        with info2:
            if st.button("🔄 Reset", key="reset_btn"):
                reset_all()
                st.rerun()

        analyze = st.button(
            "🚀 Analyze Resume",
            key="analyze_btn",
            use_container_width=True,
            disabled=st.session_state.file_bytes is None,
        )

        if analyze:
            with st.spinner("🔍 Extracting text and predicting..."):
                extracted = extract_text_router(
                    st.session_state.file_name,
                    st.session_state.file_bytes,
                )

                cleaned = clean_text(extracted)

                if len(cleaned.strip()) < 30:
                    # show extra hint for DOC files if textract missing
                    ext = Path(st.session_state.file_name).suffix.lower()
                    if ext == ".doc":
                        st.error(
                            "❌ Could not extract readable text from this .DOC file.\n\n"
                            "Tip: convert it to **.DOCX** and try again.\n"
                            "If you want .DOC support, install **textract** and system dependencies."
                        )
                    else:
                        st.error(
                            "❌ Could not extract readable text.\n\n"
                            "Tip: Try a different file, or ensure the PDF is not scanned-image only."
                        )
                else:
                    pred_label, top3 = predict_label(cleaned)

                    st.session_state.extracted_text = extracted
                    st.session_state.cleaned_text = cleaned
                    st.session_state.prediction = pred_label
                    st.session_state.top3 = top3
                    st.session_state.processed = True
                    st.rerun()

        # Results
        if st.session_state.processed and st.session_state.prediction:
            st.markdown("---")

            st.markdown(
                f"""
                <div class="resultBox">
                    <div class="resultLabel">✅ Predicted Category</div>
                    <div class="resultValue">{st.session_state.prediction}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence block (if available)
            if st.session_state.top3:
                st.write("")
                st.markdown("**🎯 Top Predictions**")
                for i, (lbl, p) in enumerate(st.session_state.top3, start=1):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.write(f"**{i}. {lbl}**")
                        st.progress(min(max(p, 0.0), 1.0))
                    with c2:
                        st.write(f"**{p*100:.1f}%**")
            else:
                st.caption("Confidence not available (model has no predict_proba).")

            # Preview tabs
            st.write("")
            tab1, tab2 = st.tabs(["📝 Cleaned Text Preview", "📄 Raw Extracted Preview"])
            with tab1:
                st.text_area(
                    "Cleaned Text (used by model)",
                    value=st.session_state.cleaned_text[:8000],
                    height=260,
                    label_visibility="collapsed",
                )
            with tab2:
                st.text_area(
                    "Raw Extracted Text",
                    value=st.session_state.extracted_text[:8000],
                    height=260,
                    label_visibility="collapsed",
                )

            # Download report
            st.write("")
            report = (
                f"Resume: {st.session_state.file_name}\n"
                f"Predicted Category: {st.session_state.prediction}\n\n"
            )
            if st.session_state.top3:
                report += "Top-3 Predictions:\n"
                for lbl, p in st.session_state.top3:
                    report += f"- {lbl}: {p*100:.2f}%\n"
                report += "\n"

            report += "Cleaned Text:\n" + st.session_state.cleaned_text + "\n"

            st.download_button(
                "📥 Download Prediction Report (.txt)",
                data=report,
                file_name=f"prediction_report_{Path(st.session_state.file_name).stem}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    else:
        st.markdown(
            "<div class='muted'>👆 Upload a resume file to begin. Supported: PDF, DOCX, TXT. (.DOC is best-effort)</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# Footer info
st.write("")
st.caption(
    "Tip: If your PDF is scanned (image-only), text extraction may fail. Convert to text-based PDF or DOCX for best results."
)
