import queue, re, subprocess, sys, wave
import numpy as np
import sounddevice as sd
import ollama
from faster_whisper import WhisperModel

DEVICE = None
SR = 16000
CHUNK_SEC = 3
LLM_MODEL = "llama3.2:1b"
VOICE = "en_US-lessac-medium"
MAX_WAIT_SEC = 15

EXIT_PHRASES = ["exit program", "stop program", "quit program", "stop listening"]
OK_WORDS = ["ok", "okay", "thank you", "thanks", "got it", "understood"]

HAZARDS = {
    "FIRE": {
        "words": ["fire", "fires", "smoke", "burning", "flames", "blaze"],
        "steps": "Raise the alarm and warn others. Use an extinguisher only if the fire is small and you have a clear exit behind you. Otherwise go to the nearest fire exit or stairs. Do not use the lift. Stay low if there is smoke. Go to the assembly point and report to the supervisor.",
    },
    "GAS LEAK": {
        "words": ["gas", "leak", "fumes", "smell"],
        "steps": "Stop work. Do not switch anything on or off. No flames, sparks or phones near the area. Move to fresh air, upwind. Warn others and inform the supervisor.",
    },
    "FALL OR INJURY": {
        "words": ["fell", "fall", "injured", "bleeding", "unconscious", "hurt"],
        "steps": "Do not move the person unless there is danger. Call first aid and emergency services. Control bleeding with firm pressure. Keep the person calm and warm until help arrives.",
    },
    "ELECTRICAL HAZARD": {
        "words": ["shock", "electric", "electrocuted", "sparks", "wire"],
        "steps": "Do not touch the person or the wire. Switch off the power at the main if it is safe. Keep everyone away. Call emergency services and first aid.",
    },
    "COLLAPSE": {
        "words": ["collapse", "crack", "scaffold", "trapped"],
        "steps": "Move away from the structure immediately. Warn everyone nearby. Do not go back in. Call emergency services and the supervisor.",
    },
    "WELDING": {
        "words": ["welding", "weld", "torch"],
        "steps": "Wear a welding helmet, gloves and protective clothing. Remove flammable materials from the area. Keep a fire extinguisher nearby. Make sure there is ventilation. Secure the gas cylinders. Assign a fire watch.",
    },
}

SYSTEM_PROMPT = (
    "You are a construction site safety assistant. A worker reported the "
    "situation below. The detected hazard is: {hazard}. Give two short extra "
    "safety tips that are not obvious. Plain sentences only, no numbers, "
    "bullets or symbols. Maximum 40 words."
)

print("Loading speech model...")
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(indata[:, 0].copy())

def drain():
    while not q.empty():
        q.get()

def clean(text):
    text = re.sub(r"[*#_`>\-]+", " ", text)
    text = re.sub(r"\b\d+[.)]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()

def speak(text):
    text = clean(text)
    if not text:
        return
    print("AI:", text)
    subprocess.run(
        [sys.executable, "-m", "piper", "-m", VOICE,
         "--data-dir", "voices", "-f", "out.wav"],
        input=text.encode("utf-8"), check=True)
    with wave.open("out.wav", "rb") as w:
        sr = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    sd.play(audio, sr)
    sd.wait()
    drain()

def get_chunk():
    buf, n = [], 0
    while n < SR * CHUNK_SEC:
        data = q.get()
        buf.append(data)
        n += len(data)
    return np.concatenate(buf)

def listen():
    drain()
    spoken, waited = [], 0
    while True:
        audio = get_chunk()
        segs, _ = model.transcribe(
            audio, language="en", beam_size=3,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            initial_prompt="Construction safety. Fire in the building, gas leak, "
                           "someone fell, electric shock, welding, collapse, okay.")
        text = " ".join(s.text.strip() for s in segs)
        if text:
            spoken.append(text)
        elif spoken:
            break
        else:
            waited += CHUNK_SEC
            if waited >= MAX_WAIT_SEC:
                break
    return " ".join(spoken)

def detect_hazard(text):
    words = text.lower().replace(".", " ").replace(",", " ").split()
    for name, info in HAZARDS.items():
        if any(w in words for w in info["words"]):
            return name
    return None

def is_ok(text):
    t = " " + re.sub(r"[^a-z ]", "", text.lower()).strip() + " "
    return len(t.split()) <= 4 and any(" " + w + " " in t for w in OK_WORDS)

def speak_llm(text, hazard):
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT.format(hazard=hazard)},
                  {"role": "user", "content": text}],
        stream=True,
        options={"num_predict": 90, "temperature": 0.2})
    buf = ""
    for chunk in stream:
        buf += chunk["message"]["content"]
        parts = re.split(r"(?<=[.!?])\s+", buf)
        for sentence in parts[:-1]:
            if sentence.strip():
                speak(sentence)
        buf = parts[-1]
    if buf.strip():
        speak(buf)

def main():
    with sd.InputStream(device=DEVICE, samplerate=SR, channels=1,
                        dtype="float32", callback=callback):
        speak("Safety assistant ready. What is the situation?")   # asked only once
        while True:
            print("Listening...")
            text = listen()
            if not text:
                continue
            print("You:", text)

            if any(p in text.lower() for p in EXIT_PHRASES):
                speak("Stopping. Stay safe.")
                return

            if is_ok(text):
                print("OK heard. Standing by silently.")
                continue

            hazard = detect_hazard(text)
            if not hazard:
                continue

            print(f"\n*** ALERT: {hazard} ***")
            speak(f"{hazard.title()} detected.")
            speak(HAZARDS[hazard]["steps"])
            speak_llm(text, hazard)
            print("-" * 40)

try:
    main()
except KeyboardInterrupt:
    print("\nStopped.")