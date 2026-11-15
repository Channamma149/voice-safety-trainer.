import queue
import numpy as np
import sounddevice as sd
import ollama
from faster_whisper import WhisperModel

DEVICE = None
SR = 16000
CHUNK_SEC = 3
LLM_MODEL = "llama3.2:1b"

# ---- Keyword -> instant safety instructions ----
HAZARDS = {
    "FIRE": {
        "words": ["fire", "smoke", "burning", "flame", "blaze"],
        "steps": "1) Raise the alarm and warn others. 2) Use a fire extinguisher ONLY if the fire is small and you have a clear exit behind you. 3) Otherwise go to the nearest fire exit or stairs. 4) Do NOT use the lift. 5) Stay low if there is smoke. 6) Go to the assembly point and report to the supervisor.",
    },
    "GAS LEAK": {
        "words": ["gas", "leak", "fumes"],
        "steps": "1) Stop work and do not switch anything on or off. 2) No flames, sparks, or phones near the area. 3) Move to fresh air, upwind. 4) Warn others and inform the supervisor.",
    },
    "FALL / INJURY": {
        "words": ["fell", "fall", "injured", "bleeding", "unconscious", "hurt"],
        "steps": "1) Do not move the person unless there is danger. 2) Call for first aid and emergency services. 3) Control bleeding with firm pressure. 4) Keep the person calm and warm until help arrives.",
    },
    "ELECTRICAL": {
        "words": ["shock", "electric", "electrocuted", "sparks", "wire"],
        "steps": "1) Do NOT touch the person or wire directly. 2) Switch off the power at the main if safe. 3) Keep everyone away. 4) Call emergency services and first aid.",
    },
    "COLLAPSE": {
        "words": ["collapse", "crack", "scaffold", "falling", "trapped"],
        "steps": "1) Move away from the structure immediately. 2) Warn everyone nearby. 3) Do not re-enter. 4) Call emergency services and the supervisor.",
    },
    "FLOOD / EXPLOSION": {
        "words": ["flood", "water", "explosion", "blast"],
        "steps": "1) Move to higher or safer ground. 2) Keep away from electrical equipment. 3) Evacuate to the assembly point and inform the supervisor.",
    },
}

SYSTEM_PROMPT = (
    "You are a construction site safety assistant. A worker spoke the text below. "
    "The detected hazard is: {hazard}. Give exactly 3 short, numbered safety "
    "instructions for the worker, such as finding the nearest fire exit, using the "
    "right equipment, or moving to another floor. Be calm and direct. "
    "Maximum 50 words. No introduction."
)

model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(indata[:, 0].copy())

def detect_hazard(text):
    t = text.lower()
    for name, info in HAZARDS.items():
        if any(w in t for w in info["words"]):
            return name
    return None

def ask_llm(text, hazard):
    r = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(hazard=hazard)},
            {"role": "user", "content": text},
        ],
        options={"num_predict": 120, "temperature": 0.2},
    )
    return r["message"]["content"].strip()

with sd.InputStream(device=DEVICE, samplerate=SR, channels=1,
                    dtype="float32", callback=callback):
    print("Listening... press Ctrl+C to stop")
    buf, n = [], 0
    while True:
        data = q.get()
        buf.append(data)
        n += len(data)
        if n >= SR * CHUNK_SEC:
            audio = np.concatenate(buf)
            buf, n = [], 0
            segments, _ = model.transcribe(
                audio, language="en", beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments)
            if not text:
                continue
            print("\nYou:", text)

            hazard = detect_hazard(text)
            if hazard:
                print(f"\n*** ALERT: {hazard} ***")
                print("Immediate steps:", HAZARDS[hazard]["steps"])
                print("\nAI advice:", ask_llm(text, hazard))
                print("-" * 40)