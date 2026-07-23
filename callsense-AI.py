# %%
!pip install assemblyai groq python-dotenv pandas --quiet

# %%
import os
from dotenv import load_dotenv
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

assert ASSEMBLYAI_API_KEY, "Missing ASSEMBLYAI_API_KEY — add it to your .env file"
assert GROQ_API_KEY, "Missing GROQ_API_KEY — add it to your .env file"

print("Keys loaded OK")

# %%
import assemblyai as aai

aai.settings.api_key = ASSEMBLYAI_API_KEY

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

# %%
# Heuristic for v1: the rep almost always speaks first (answers/opens the call)
# on an outbound or inbound collections call. This is a simple assumption —
# swap in a smarter rule later (e.g. rep says a company name / script phrase first).

def label_speakers(utterances: list[dict]) -> list[dict]:
    if not utterances:
        return utterances

    first_speaker = utterances[0]["speaker"]
    role_map = {first_speaker: "rep"}

    # any other speaker label found gets mapped to "customer"
    for u in utterances:
        if u["speaker"] not in role_map:
            role_map[u["speaker"]] = "customer"

    for u in utterances:
        u["role"] = role_map[u["speaker"]]

    return utterances

# %%
# type = "keyword"  -> fast regex/substring match, good for greeting/closing phrases
# type = "semantic"  -> sent to the LLM for a yes/no judgment, good for fuzzy criteria

qa_checklist = [
    {
        "id": "greeting",
        "type": "keyword",
        "role": "rep",
        "description": "Rep greeted the customer at the start of the call",
        "keywords": ["hello", "good morning", "good afternoon", "good evening", "thank you for calling", "thanks for calling"],
        "window": "first",   # check only the rep's first utterance
    },
    {
        "id": "identity_verification",
        "type": "semantic",
        "role": "rep",
        "description": "Rep verified the caller's identity (e.g. asked for date of birth, account number, or similar identifying details)",
    },
    {
        "id": "reason_for_call",
        "type": "semantic",
        "role": "rep",
        "description": "Rep clearly explained the reason for the call (e.g. outstanding balance, claim denial, payment reminder)",
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
        "keywords": ["thank you for your time", "have a great day", "have a good day", "is there anything else", "take care"],
        "window": "last",   # check only the rep's last utterance
    },
]

# %%
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

# %%
from groq import Groq

groq_client = Groq(api_key=GROQ_API_KEY)

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

# %%
import pandas as pd

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

    df = pd.DataFrame(results)
    return df


def score_summary(df: pd.DataFrame) -> str:
    total = len(df)
    passed = df["passed"].sum()
    pct = round(100 * passed / total, 1) if total else 0
    return f"Score: {passed}/{total} checklist items passed ({pct}%)"

# %%
# Replace this path with your own uploaded .mp3 / .wav / .m4a file
AUDIO_PATH = "sample_call.mp3"   # <-- put your file here

transcript = transcribe_call(AUDIO_PATH)
utterances = transcript_to_utterances(transcript)
utterances = label_speakers(utterances)

report_df = run_qa_analysis(utterances, qa_checklist)

print(score_summary(report_df))
report_df

# %%
transcript_df = pd.DataFrame(utterances)
transcript_df

# %%
print(score_summary(report_df))

# %%
report_df

# %%



