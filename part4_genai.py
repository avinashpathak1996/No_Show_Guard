import os
import json
import time
import getpass
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field, field_validator, ValidationError
from google import genai
from google.genai import types

# ==========================================
# SETUP: API Key & Model Config
# ==========================================
# Safe runtime input so the key is never saved in your code/GitHub
API_KEY = getpass.getpass("Paste your Google AI Studio API Key (input hidden): ").strip()

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        timeout=180000,
        retry_options=types.HttpRetryOptions(
            attempts=5
        )
    )
)

# Auto-detect available Flash models for this API key

available_models = [
    m.name.replace("models/", "")
    for m in client.models.list()
    if "flash" in m.name.lower()
    and "generateContent" in (m.supported_actions or [])
]

print("Available Flash Models:", available_models)

# Preferred model selection
PREFERRED_MODELS = [
    "gemini-3.8-flash"
]

MODEL_NAME = next(
    (model for model in PREFERRED_MODELS
     if model in available_models),
    None
)

if MODEL_NAME is None:
    raise RuntimeError(
        "No supported preferred Gemini Flash model found. "
        "Please check your API key and available models."
    )

print("Using Model:", MODEL_NAME)
print("Testing Model:", MODEL_NAME)
print("Model repr:", repr(MODEL_NAME))
# QUICK API TEST

print("\n--- QUICK API TEST ---")

try:
    test_response = client.models.generate_content(
        model=MODEL_NAME,
        contents="Reply with only: OK",
        config=types.GenerateContentConfig(
            max_output_tokens=100
        )
    )

    print("Full API Response:", test_response)

    if test_response.text:
        print("Quick test response:", test_response.text)
    else:
        print("Response received, but text is empty.")
        print("Candidates:", test_response.candidates)
        print("Prompt Feedback:", test_response.prompt_feedback)

except Exception as e:
    print("Quick test failed:", repr(e))

# raise SystemExit

ALLOWED_ROOT_CAUSES = [
    "forgot_or_no_reminder",
    "distance_or_transport",
    "cost_or_fee",
    "illness_or_emergency",
    "work_or_schedule",
    "other"
]
ALLOWED_ROOT_CAUSES = [
    "forgot_or_no_reminder",
    "distance_or_transport",
    "cost_or_fee",
    "illness_or_emergency",
    "work_or_schedule",
    "other"
]

SYSTEM_TEXT = (
    "You are a clinical operations analyst extracting structured reasons from patient follow-up call notes. "
    "Always return valid JSON matching the exact schema requested."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string"},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
        "recommended_action": {"type": "string"}
    },
    "required": ["root_cause", "confidence", "summary", "recommended_action"]
}

api_call_count = 0

def build_craft_prompt(note_text: str) -> str:
    return f"""
Context: A healthcare clinic front desk called patients who missed their scheduled appointments and logged short notes (some in English, some in Hinglish, some ambiguous).
Role: Act as a precise medical operations data extractor.
Action: Read the follow-up note below and classify the primary reason the patient missed their appointment. Choose ONLY from these exact allowed values for root_cause:
{ALLOWED_ROOT_CAUSES}
If the note is ambiguous or unclear, classify root_cause as "other" with lower confidence rather than guessing.
Tone: Objective, concise, and operational.
Format: Return a JSON object with:
- root_cause: exact string from the allowed list
- confidence: float between 0 and 1
- summary: concise summary of the note (at most 25 words)
- recommended_action: concrete next step for the clinic front desk

Follow-up Note: "{note_text}"
"""

def analyse_note(note_text: str, custom_prompt: str = None) -> tuple[dict, str]:
    global api_call_count
    prompt_to_use = custom_prompt if custom_prompt is not None else build_craft_prompt(note_text)
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_TEXT,
        temperature=0,
        max_output_tokens=500,
        response_mime_type="application/json",
        response_schema=SCHEMA
    )
    

    print("Sending request to Gemini API...")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_to_use,
            config=config
        )

        print("Gemini API response received!")

        raw_text = response.text

        if not raw_text:
            raise ValueError("Gemini returned an empty response.")

        data = json.loads(raw_text)
        api_call_count += 1

        return data, raw_text

    except Exception as e:
        print("API / Processing Error:", repr(e))
        raise


# ==========================================
# TASK 4.1: One Validated Call (5 Marks)
# ==========================================
print("="*60)
print("TASK 4.1: ONE VALIDATED CALL ON NOTE N001")
print("="*60)

notes_df = pd.read_csv("Data/followup_notes.csv")
n001_text = notes_df.loc[notes_df["note_id"] == "N001", "note_text"].values[0]

n001_data, _ = analyse_note(n001_text)
print("Parsed Dictionary for N001:")
print(json.dumps(n001_data, indent=2))


# ==========================================
# TASK 4.2: Validate & Process First 10 Notes (7 Marks)
# ==========================================
print("\n" + "="*60)
print("TASK 4.2: PROCESSING FIRST 10 NOTES WITH PYDANTIC")
print("="*60)

class FollowupInsight(BaseModel):
    root_cause: str = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=5)
    recommended_action: str = Field(min_length=5)

    @field_validator("root_cause")
    @classmethod
    def check_allowed_cause(cls, v: str) -> str:
        if v not in ALLOWED_ROOT_CAUSES:
            raise ValueError(f"root_cause '{v}' is not in allowed list")
        return v

results_rows = []
first_10 = notes_df.head(10)

for idx, row in first_10.iterrows():
    note_id = row["note_id"]
    appt_id = row["appointment_id"]
    raw_note = row["note_text"]
    
    # Input validation: check NaN first, then empty/whitespace string
    if pd.isna(raw_note) or str(raw_note).strip() == "":
        print(f"[{note_id}] Blank note detected -> Skipped API call.")
        results_rows.append({
            "note_id": note_id,
            "appointment_id": appt_id,
            "root_cause": "other",
            "confidence": 0.0,
            "summary": "No note recorded",
            "recommended_action": "Re-call patient",
            "validation_status": "skipped_blank",
            "raw_response": ""
        })
        continue
    
    # Pause 13 seconds between API calls to respect free-tier rate limit
    print(f"[{note_id}] Calling Gemini API (waiting 13s for rate limit)...")
    time.sleep(13)
    
    raw_reply = ""
    try:
        parsed_dict, raw_reply = analyse_note(str(raw_note).strip())
        # Validate with Pydantic
        insight = FollowupInsight(**parsed_dict)
        results_rows.append({
            "note_id": note_id,
            "appointment_id": appt_id,
            "root_cause": insight.root_cause,
            "confidence": insight.confidence,
            "summary": insight.summary,
            "recommended_action": insight.recommended_action,
            "validation_status": "valid",
            "raw_response": raw_reply
        })
        print(f"    -> Valid! root_cause = {insight.root_cause} (conf={insight.confidence})")
    except Exception as e:
        print(f"    -> Invalid reply caught: {e}")
        # Keep raw reply without repairing; copy any fields present, leave missing empty
        safe_dict = parsed_dict if isinstance(locals().get("parsed_dict"), dict) else {}
        results_rows.append({
            "note_id": note_id,
            "appointment_id": appt_id,
            "root_cause": safe_dict.get("root_cause", ""),
            "confidence": safe_dict.get("confidence", ""),
            "summary": safe_dict.get("summary", ""),
            "recommended_action": safe_dict.get("recommended_action", ""),
            "validation_status": "invalid",
            "raw_response": raw_reply
        })

notes_analysed_df = pd.DataFrame(results_rows)
notes_analysed_df.to_csv("notes_analysed.csv", index=False)
print("\nSaved 'notes_analysed.csv' successfully! Shape:", notes_analysed_df.shape)
print(notes_analysed_df[["note_id", "root_cause", "confidence", "validation_status"]].to_string(index=False))


# ==========================================
# TASK 4.3: Break It On Purpose (5 Marks)
# ==========================================
print("\n" + "="*60)
print("TASK 4.3: BREAK IT ON PURPOSE & PROMPT COMPARISON")
print("="*60)

# (a) Feed two deliberately bad replies to Pydantic
bad_reply_1 = {
    "root_cause": "forgot_or_no_reminder",
    "confidence": 1.7,  # > 1.0 limit
    "summary": "Patient forgot appointment.",
    "recommended_action": "Send SMS reminder"
}
bad_reply_2 = {
    "root_cause": "cost_or_fee",
    "confidence": 0.85,
    "summary": "Fee was too high for patient."
    # missing recommended_action
}

print("(a) Testing Deliberately Bad Replies:")
for i, bad_data in enumerate([bad_reply_1, bad_reply_2], 1):
    try:
        FollowupInsight(**bad_data)
    except ValidationError as ve:
        print(f"    Bad Reply {i} caught cleanly -> {ve.errors()[0]['msg']}")

# (b) Compare V1 ("Classify this note") vs V2 (CRAFT prompt) on N007 and N009
print("\n(b) Comparing Prompt V1 vs Prompt V2 on N007 (Hinglish) and N009 (Ambiguous):")
comparison_rows = []

for target_id in ["N007", "N009"]:
    txt = notes_df.loc[notes_df["note_id"] == target_id, "note_text"].values[0]
    
    # Run V1
    time.sleep(13)
    v1_data, _ = analyse_note(txt, custom_prompt=f'Classify this note: "{txt}"')
    v1_allowed = v1_data.get("root_cause") in ALLOWED_ROOT_CAUSES
    comparison_rows.append({
        "Note": target_id,
        "Prompt": "V1 (Naive)",
        "root_cause": v1_data.get("root_cause"),
        "confidence": v1_data.get("confidence"),
        "In_Allowed_List": v1_allowed
    })
    
    # Run V2 (reuse from 4.2 or run fresh)
    v2_row = notes_analysed_df[notes_analysed_df["note_id"] == target_id].iloc[0]
    comparison_rows.append({
        "Note": target_id,
        "Prompt": "V2 (CRAFT)",
        "root_cause": v2_row["root_cause"],
        "confidence": v2_row["confidence"],
        "In_Allowed_List": v2_row["root_cause"] in ALLOWED_ROOT_CAUSES
    })

comp_table = pd.DataFrame(comparison_rows)
print(comp_table.to_string(index=False))
print("Deployment Choice: Deploy V2 (CRAFT) because it strictly enforces the allowed root_cause taxonomy and avoids hallucinating categories on Hinglish/ambiguous notes.")

# (c) Total API calls across Tasks 4.1 - 4.3
print(f"\n(c) Total Gemini API calls used across Tasks 4.1 - 4.3: {api_call_count}")


# ==========================================
# TASK 4.4: Merge & Cross-Tabulate (3 Marks)
# ==========================================
print("\n" + "="*60)
print("TASK 4.4: MERGE & CROSS-TABULATION")
print("="*60)

clean_df = pd.read_csv("appointments_clean.csv")
merged_notes = pd.merge(notes_analysed_df, clean_df, on="appointment_id", how="inner")
print(f"Merged table rows: {len(merged_notes)}")

ct = pd.crosstab(merged_notes["root_cause"], merged_notes["reminder_sent"], margins=True)
print("\nCross-tabulation (root_cause vs reminder_sent):")
print(ct)