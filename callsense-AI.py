import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import assemblyai as aai
from groq import Groq

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(page_title="CallSense AI", page_icon="📞", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% -10%, #0f2547 0%, #050b16 55%, #02040a 100%);
        color: #e8f1ff;
    }
    #MainMenu, footer, header {visibility: hidden;}

    .hero-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 28px;
        padding: 18px 0 6px 0;
        flex-wrap: wrap;
    }
    .hero-text h1 {
        font-size: 2.3rem;
        margin-bottom: 2px;
        background: linear-gradient(90deg, #5ce1e6, #7cf29c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .hero-text p {
        color: #9fb3cc;
        font-size: 0.98rem;
        margin-top: 0;
    }
    .badge-row { margin-top: 6px; }
    .badge {
        display: inline-block;
        font-size: 0.72rem;
        padding: 3px 10px;
        margin-right: 6px;
        border-radius: 999px;
        background: rgba(92,225,230,0.12);
        border: 1px solid rgba(92,225,230,0.35);
        color: #5ce1e6;
    }

    /* --- Robot --- */
    .robot-box { width: 130px; height: 130px; position: relative; }
    .robot-head {
        width: 84px; height: 70px;
        margin: 0 auto;
        background: linear-gradient(145deg, #1c2f4a, #0f1c30);
        border: 2px solid #5ce1e6;
        border-radius: 18px;
        position: relative;
        box-shadow: 0 0 24px rgba(92,225,230,0.35);
        animation: bob 2.6s ease-in-out infinite;
    }
    .robot-antenna {
        width: 3px; height: 16px; background: #5ce1e6;
        margin: 0 auto; position: relative; top: -16px;
    }
    .robot-antenna::after {
        content: ""; position: absolute; top: -8px; left: -3px;
        width: 9px; height: 9px; border-radius: 50%;
        background: #7cf29c;
        box-shadow: 0 0 10px #7cf29c;
        animation: pulse 1.4s ease-in-out infinite;
    }
    .robot-eyes {
        display: flex; justify-content: center; gap: 14px;
        padding-top: 22px;
    }
    .eye {
        width: 12px; height: 12px; border-radius: 50%;
        background: #5ce1e6;
        box-shadow: 0 0 10px #5ce1e6, 0 0 20px rgba(92,225,230,0.6);
        animation: blink 3.2s infinite;
    }
    .robot-cross {
        position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
        width: 16px; height: 16px; opacity: 0.85;
    }
    .robot-cross::before, .robot-cross::after {
        content: ""; position: absolute; background: #7cf29c; border-radius: 2px;
    }
    .robot-cross::before { width: 16px; height: 3px; top: 6.5px; left: 0; }
    .robot-cross::after { width: 3px; height: 16px; top: 0; left: 6.5px; }
    .robot-body {
        width: 96px; height: 34px;
        margin: 6px auto 0 auto;
        background: linear-gradient(145deg, #16283f, #0b1524);
        border: 2px solid rgba(92,225,230,0.5);
        border-radius: 12px;
    }

    @keyframes bob {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }
    @keyframes blink {
        0%, 92%, 100% { transform: scaleY(1); }
        95% { transform: scaleY(0.1); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.7); }
    })


ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not ASSEMBLYAI_API_KEY or not GROQ_API_KEY:
    st.error(
        "Missing ASSEMBLYAI_API_KEY and/or GROQ_API_KEY. "
        "Add them to a .env file in this folder before running the app."
    )
    st.stop()

aai.settings.api_key = ASSEMBLYAI_API_KEY
groq_client = Groq(api_key=GROQ_API_KEY)

# ----------------------------------------------------------------------------
# QA checklist
# type = "keyword"  -> fast regex/substring match, good for greeting/closing phrases
# type = "semantic" -> sent to the LLM for a yes/no judgment, good for fuzzy criteria
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Core pipeline functions (unchanged logic from the notebook)
# ----------------------------------------------------------------------------
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

    NOTE: AssemblyAI labels speakers as "A", "B" etc it does NOT know
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

    prompt = f"You are a call quality auditor reviewing a transcript of a REP's turns only."

Criteria to check: "{item['description']}"

Rep's utterances during the call:
{rep_text}

Answer strictly in this format:
ANSWER: yes or no
EVIDENCE: the exact quote from the rep's utterances that supports your answer (or "none" if the answer is no)
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.title("📞 CallSense AI — Call QA Analyzer")
st.caption("Upload a call recording to check it against the QA checklist.")

with st.sidebar:
    st.header("QA Checklist")
    for item in qa_checklist:
        st.markdown(f"- **{item['id']}** ({item['type']}): {item['description']}")

uploaded_file = st.file_uploader(
    "Upload a call recording",
    type=["mp3", "wav", "m4a", "mp4"],
    help="The rep is assumed to be whoever speaks first in the recording.",
)

analyze_clicked = st.button("Analyze call", type="primary", disabled=uploaded_file is None)

if analyze_clicked and uploaded_file is not None:
    # Save the uploaded file to a temp path AssemblyAI can read from disk
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
        st.metric("QA Score", f"{passed}/{total} passed", f"{pct}%")

        st.dataframe(
            report_df.rename(columns={
                "checklist_id": "Check",
                "description": "Description",
                "passed": "Passed",
                "evidence": "Evidence",
            }),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View full transcript"):
            transcript_df = pd.DataFrame(utterances)
            st.dataframe(
                transcript_df.rename(columns={
                    "speaker": "Speaker",
                    "text": "Text",
                    "start_ms": "Start (ms)",
                    "end_ms": "End (ms)",
                    "role": "Role",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download QA report (CSV)",
            report_df.to_csv(index=False).encode("utf-8"),
            file_name="qa_report.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")

    finally:
        os.remove(audio_path)

elif uploaded_file is None:
    st.info("Upload an audio file above, then click **Analyze call** to run the QA check.")
