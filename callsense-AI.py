import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import assemblyai as aai
from groq import Groq

# ==============================================================================
# App configuration
# ==============================================================================
load_dotenv()

st.set_page_config(
    page_title="CallSense AI",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not ASSEMBLYAI_API_KEY or not GROQ_API_KEY:
    st.error(
        "Missing **ASSEMBLYAI_API_KEY** and/or **GROQ_API_KEY**. "
        "Add them to a `.env` file in this folder before running the app."
    )
    st.stop()

aai.settings.api_key = ASSEMBLYAI_API_KEY
groq_client = Groq(api_key=GROQ_API_KEY)

LLM_MODEL = "llama-3.3-70b-versatile"

# ==============================================================================
# Theming — custom CSS
# ==============================================================================
CUSTOM_CSS = """
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    /* ---- Hero header ---- */
    .cs-hero {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-bottom: 0.25rem;
    }
    .cs-hero-icon {
        font-size: 2.1rem;
        line-height: 1;
    }
    .cs-hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #101828;
    }
    .cs-hero-subtitle {
        color: #667085;
        font-size: 1rem;
        margin: 0.15rem 0 1.6rem 0;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #101828;
    }
    section[data-testid="stSidebar"] * {
        color: #E4E7EC !important;
    }
    .cs-sidebar-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .cs-check-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 10px;
        padding: 0.65rem 0.8rem;
        margin-bottom: 0.55rem;
    }
    .cs-check-card .cs-check-name {
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 0.15rem;
    }
    .cs-check-card .cs-check-desc {
        font-size: 0.78rem;
        color: #98A2B3 !important;
        line-height: 1.35;
    }
    .cs-badge {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.1rem 0.5rem;
        border-radius: 999px;
        margin-bottom: 0.35rem;
    }
    .cs-badge-keyword { background: #1D2939; color: #A6F4C5 !important; }
    .cs-badge-semantic { background: #1D2939; color: #B2CCFF !important; }

    /* ---- Upload card ---- */
    .cs-panel {
        background: #FFFFFF;
        border: 1px solid #EAECF0;
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        margin-bottom: 1.4rem;
    }

    /* ---- Score summary ---- */
    .cs-score-wrap {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .cs-score-card {
        flex: 1;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        border: 1px solid #EAECF0;
        background: #FFFFFF;
    }
    .cs-score-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #667085;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }
    .cs-score-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #101828;
    }
    .cs-score-value.good { color: #067647; }
    .cs-score-value.warn { color: #B54708; }
    .cs-score-value.bad { color: #B42318; }

    /* ---- Result rows ---- */
    .cs-result-row {
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        border: 1px solid #EAECF0;
        margin-bottom: 0.6rem;
        background: #FFFFFF;
    }
    .cs-result-row.pass { border-left: 4px solid #12B76A; }
    .cs-result-row.fail { border-left: 4px solid #F04438; }
    .cs-result-icon { font-size: 1.15rem; margin-top: 0.05rem; }
    .cs-result-title { font-weight: 700; color: #101828; font-size: 0.95rem; }
    .cs-result-desc { color: #667085; font-size: 0.85rem; margin-top: 0.1rem; }
    .cs-result-evidence {
        margin-top: 0.45rem;
        font-size: 0.82rem;
        font-style: italic;
        color: #475467;
        background: #F9FAFB;
        border-radius: 8px;
        padding: 0.5rem 0.7rem;
        border: 1px dashed #D0D5DD;
    }

    /* ---- Transcript bubbles ---- */
    .cs-bubble-row { display: flex; margin-bottom: 0.55rem; }
    .cs-bubble-row.rep { justify-content: flex-start; }
    .cs-bubble-row.customer { justify-content: flex-end; }
    .cs-bubble {
        max-width: 72%;
        padding: 0.55rem 0.85rem;
        border-radius: 14px;
        font-size: 0.87rem;
        line-height: 1.4;
    }
    .cs-bubble.rep { background: #EFF4FF; color: #1D2939; border-bottom-left-radius: 3px; }
    .cs-bubble.customer { background: #F2F4F7; color: #1D2939; border-bottom-right-radius: 3px; }
    .cs-bubble-speaker {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.15rem;
        opacity: 0.6;
    }

    .stButton>button {
        border-radius: 9px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# QA checklist
# type = "keyword"  -> fast regex/substring match, good for greeting/closing phrases
# type = "semantic" -> sent to the LLM for a yes/no judgment, good for fuzzy criteria
# ==============================================================================
qa_checklist = [
    {
        "id": "greeting",
        "type": "keyword",
        "role": "rep",
        "description": "Rep greeted the customer at the start of the call",
        "keywords": [
            "hello", "good morning", "good afternoon", "good evening",
            "thank you for calling", "thanks for calling",
        ],
        "window": "first",  # check only the rep's first utterance
    },
    {
        "id": "identity_verification",
        "type": "semantic",
        "role": "rep",
        "description": (
            "Rep verified the caller's identity (e.g. asked for date of birth, "
            "account number, or similar identifying details)"
        ),
    },
    {
        "id": "reason_for_call",
        "type": "semantic",
        "role": "rep",
        "description": (
            "Rep clearly explained the reason for the call (e.g. outstanding "
            "balance, claim denial, payment reminder)"
        ),
    },
    {
        "id": "asked_required_question",
        "type": "semantic",
        "role": "rep",
        "description": "Rep asked the customer about their preferred payment date or payment plan",
    },
    {
        "id": "closing",
        "type": "keyword",
        "role": "rep",
        "description": "Rep closed the call properly (e.g. thanked the customer, confirmed next steps)",
        "keywords": [
            "thank you for your time", "have a great day", "have a good day",
            "is there anything else", "take care",
        ],
        "window": "last",  # check only the rep's last utterance
    },
]

CHECKLIST_LABELS = {
    "greeting": "Greeting",
    "identity_verification": "Identity Verification",
    "reason_for_call": "Reason For Call",
    "asked_required_question": "Payment Preference Asked",
    "closing": "Proper Closing",
}


# ==============================================================================
# Core pipeline functions (logic unchanged)
# ==============================================================================
def transcribe_call(audio_path: str):
    """
    Transcribes an audio file with speaker diarization.
    Returns the AssemblyAI transcript object (has .utterances with speaker labels).
    """
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return transcript


def transcript_to_utterances(transcript) -> list[dict]:
    """
    Converts the transcript into a clean list of turns:
    [{"speaker": "A", "text": "...", "start_ms": ..., "end_ms": ...}, ...]

    NOTE: AssemblyAI labels speakers as "A", "B" etc — it does NOT know
    which one is the rep and which is the customer. We infer that next.
    """
    return [
        {
            "speaker": u.speaker,
            "text": u.text,
            "start_ms": u.start,
            "end_ms": u.end,
        }
        for u in transcript.utterances
    ]


def label_speakers(utterances: list[dict]) -> list[dict]:
    """
    Heuristic for v1: the rep almost always speaks first (answers/opens the call)
    on an outbound or inbound collections call. Swap in a smarter rule later
    (e.g. rep says a company name / script phrase first).
    """
    if not utterances:
        return utterances

    first_speaker = utterances[0]["speaker"]
    role_map = {first_speaker: "rep"}

    for u in utterances:
        if u["speaker"] not in role_map:
            role_map[u["speaker"]] = "customer"

    for u in utterances:
        u["role"] = role_map[u["speaker"]]

    return utterances


def run_keyword_check(item: dict, rep_utterances: list[dict]) -> dict:
    if not rep_utterances:
        return {"id": item["id"], "passed": False, "evidence": None}

    if item.get("window") == "first":
        candidates = [rep_utterances[0]]
    elif item.get("window") == "last":
        candidates = [rep_utterances[-1]]
    else:
        candidates = rep_utterances

    for u in candidates:
        text_lower = u["text"].lower()
        for kw in item["keywords"]:
            if kw in text_lower:
                return {"id": item["id"], "passed": True, "evidence": u["text"]}

    return {"id": item["id"], "passed": False, "evidence": None}


def run_semantic_check(item: dict, rep_utterances: list[dict]) -> dict:
    rep_text = "\n".join(f"- {u['text']}" for u in rep_utterances)

    prompt = f"""You are a call quality auditor reviewing a transcript of a REP's turns only.

Criteria to check: "{item['description']}"

Rep's utterances during the call:
{rep_text}

Answer strictly in this format:
ANSWER: yes or no
EVIDENCE: the exact quote from the rep's utterances that supports your answer (or "none" if the answer is no)
"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    reply = response.choices[0].message.content.strip()

    answer_line = next((l for l in reply.splitlines() if l.upper().startswith("ANSWER")), "ANSWER: no")
    evidence_line = next((l for l in reply.splitlines() if l.upper().startswith("EVIDENCE")), "EVIDENCE: none")

    passed = "yes" in answer_line.lower()
    evidence = evidence_line.split(":", 1)[-1].strip()

    return {"id": item["id"], "passed": passed, "evidence": None if evidence.lower() == "none" else evidence}


def run_qa_analysis(utterances: list[dict], checklist: list[dict]) -> pd.DataFrame:
    rep_utterances = [u for u in utterances if u["role"] == "rep"]

    results = []
    for item in checklist:
        if item["type"] == "keyword":
            result = run_keyword_check(item, rep_utterances)
        elif item["type"] == "semantic":
            result = run_semantic_check(item, rep_utterances)
        else:
            raise ValueError(f"Unknown checklist item type: {item['type']}")

        results.append({
            "checklist_id": item["id"],
            "description": item["description"],
            "passed": result["passed"],
            "evidence": result["evidence"],
        })

    return pd.DataFrame(results)


def score_summary(df: pd.DataFrame) -> tuple[int, int, float]:
    total = len(df)
    passed = int(df["passed"].sum())
    pct = round(100 * passed / total, 1) if total else 0.0
    return passed, total, pct


def score_tier(pct: float) -> str:
    if pct >= 80:
        return "good"
    if pct >= 50:
        return "warn"
    return "bad"


# ==============================================================================
# Render helpers
# ==============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="cs-sidebar-title">📋 QA Checklist</div>', unsafe_allow_html=True)
        for item in qa_checklist:
            badge_class = "cs-badge-keyword" if item["type"] == "keyword" else "cs-badge-semantic"
            label = CHECKLIST_LABELS.get(item["id"], item["id"])
            st.markdown(
                f"""
                <div class="cs-check-card">
                    <span class="cs-badge {badge_class}">{item['type']}</span>
                    <div class="cs-check-name">{label}</div>
                    <div class="cs-check-desc">{item['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.caption("Transcription: AssemblyAI · Reasoning: Groq (Llama 3.3 70B)")


def render_score_cards(passed: int, total: int, pct: float):
    tier = score_tier(pct)
    st.markdown(
        f"""
        <div class="cs-score-wrap">
            <div class="cs-score-card">
                <div class="cs-score-label">Checks Passed</div>
                <div class="cs-score-value {tier}">{passed} / {total}</div>
            </div>
            <div class="cs-score-card">
                <div class="cs-score-label">QA Score</div>
                <div class="cs-score-value {tier}">{pct}%</div>
            </div>
            <div class="cs-score-card">
                <div class="cs-score-label">Overall Result</div>
                <div class="cs-score-value {tier}">{"PASS" if pct >= 80 else "REVIEW" if pct >= 50 else "FAIL"}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(report_df: pd.DataFrame):
    for _, row in report_df.iterrows():
        status_class = "pass" if row["passed"] else "fail"
        icon = "✅" if row["passed"] else "❌"
        label = CHECKLIST_LABELS.get(row["checklist_id"], row["checklist_id"])
        evidence_html = (
            f'<div class="cs-result-evidence">“{row["evidence"]}”</div>' if row["evidence"] else ""
        )
        st.markdown(
            f"""
            <div class="cs-result-row {status_class}">
                <div class="cs-result-icon">{icon}</div>
                <div style="flex:1;">
                    <div class="cs-result-title">{label}</div>
                    <div class="cs-result-desc">{row['description']}</div>
                    {evidence_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_transcript(utterances: list[dict]):
    for u in utterances:
        role = u.get("role", "customer")
        speaker_label = "Rep" if role == "rep" else "Customer"
        st.markdown(
            f"""
            <div class="cs-bubble-row {role}">
                <div class="cs-bubble {role}">
                    <div class="cs-bubble-speaker">{speaker_label}</div>
                    {u['text']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# Page layout
# ==============================================================================
st.markdown(
    """
    <div class="cs-hero">
        <div class="cs-hero-icon">📞</div>
        <h1 class="cs-hero-title">CallSense AI</h1>
    </div>
    <p class="cs-hero-subtitle">
        Upload a collections call recording and automatically score it against your QA checklist.
    </p>
    """,
    unsafe_allow_html=True,
)

render_sidebar()

with st.container():
    st.markdown('<div class="cs-panel">', unsafe_allow_html=True)
    col_upload, col_button = st.columns([4, 1], vertical_alignment="bottom")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a call recording",
            type=["mp3", "wav", "m4a", "mp4"],
            help="The rep is assumed to be whoever speaks first in the recording.",
            label_visibility="collapsed",
        )
    with col_button:
        analyze_clicked = st.button(
            "Analyze Call", type="primary", disabled=uploaded_file is None, use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

if analyze_clicked and uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        audio_path = tmp.name

    try:
        with st.spinner("Transcribing call (this can take a minute for longer calls)..."):
            transcript = transcribe_call(audio_path)
            utterances = transcript_to_utterances(transcript)
            utterances = label_speakers(utterances)

        with st.spinner("Running QA checklist..."):
            report_df = run_qa_analysis(utterances, qa_checklist)

        passed, total, pct = score_summary(report_df)

        st.subheader("Results")
        render_score_cards(passed, total, pct)
        render_results(report_df)

        with st.expander("📝 View full transcript"):
            render_transcript(utterances)

        st.download_button(
            "⬇️ Download QA Report (CSV)",
            report_df.to_csv(index=False).encode("utf-8"),
            file_name="qa_report.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")

    finally:
        os.remove(audio_path)

elif uploaded_file is None:
    st.info("Upload an audio file above, then click **Analyze Call** to run the QA check.")
