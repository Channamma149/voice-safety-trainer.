import json, queue, socket, subprocess, sys, threading, wave
import numpy as np
import sounddevice as sd
import ollama
from faster_whisper import WhisperModel

DEVICE = None
SR = 16000
CHUNK_SEC = 3
LLM_MODEL = "llama3.2:1b"
VOICE = "en_US-lessac-medium"
PORT_IN, PORT_OUT = 5005, 5006      # Python listens on 5005, Unity on 5006

SCENARIOS = {
    "fire": {
        "intro": "Fire on the second floor. Smoke is spreading. What do you do?",
        "actions": {
            "Raise the alarm or warn others": ["alarm", "warn", "shout", "alert"],
            "Use the extinguisher only if the fire is small": ["extinguisher"],
            "Go to the fire exit or stairs": ["exit", "stairs", "evacuate", "escape", "leave"],
            "Avoid the lift": ["not use the lift", "avoid the lift", "no lift",
                               "not the lift", "don't use the lift", "not take the lift"],
            "Go to the assembly point": ["assembly", "muster"],
            "Inform the supervisor": ["supervisor", "manager", "report"],
        },
    },
    "welding": {
        "intro": "You are about to weld on the second floor. Look around the work area. What safety steps do you take?",
        "actions": {
            "Wear a welding helmet or face shield": ["helmet", "shield", "mask", "goggles", "face"],
            "Wear gloves and protective clothing": ["gloves", "apron", "jacket", "clothing", "ppe", "protective"],
            "Remove flammable materials": ["remove", "move", "clear", "cardboard", "rag", "flammable"],
            "Keep a fire extinguisher nearby": ["extinguisher"],
            "Ensure ventilation or fume extraction": ["ventilation", "ventilate", "fumes", "fan", "extraction"],
            "Secure the gas cylinder": ["cylinder", "secure", "chain"],
            "Assign a fire watch or inform the supervisor": ["fire watch", "watch", "supervisor", "permit", "inform"],
        },
    },
}

EXIT_PHRASES = ["exit program", "stop program", "quit program", "stop listening"]

state = {"scenario": None, "unity_ip": "127.0.0.1"}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT_IN))

def send(obj):
    sock.sendto(json.dumps(obj).encode(), (state["unity_ip"], PORT_OUT))

def listen_unity():
    while True:
        data, addr = sock.recvfrom(4096)
        try:
            msg = json.loads(data.decode())
        except ValueError:
            continue
        if msg.get("type") == "scenario" and msg.get("name") in SCENARIOS:
            state["unity_ip"] = addr[0]
            state["scenario"] = msg["name"]
            print("Unity started scenario:", msg["name"])

model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(indata[:, 0].copy())

def speak(text):
    subprocess.run(
        [sys.executable, "-m", "piper", "-m", VOICE,
         "--data-dir", "voices", "-f", "out.wav"],
        input=text.encode("utf-8"), check=True)
    with wave.open("out.wav", "rb") as w:
        sr = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    sd.play(audio, sr)
    sd.wait()
    while not q.empty():            # drop mic audio recorded while speaking
        q.get()

def evaluate(name, text):
    t = text.lower()
    done, missed = [], []
    for label, words in SCENARIOS[name]["actions"].items():
        (done if any(w in t for w in words) else missed).append(label)
    score = round(100 * len(done) / len(SCENARIOS[name]["actions"]))
    return score, done, missed

def coach(text, missed):
    prompt = ("You are a construction safety trainer. The worker said: '" + text +
              "'. They missed: " + "; ".join(missed) +
              ". In two short plain sentences, tell them what to do next time. "
              "No lists, numbers or symbols.")
    r = ollama.chat(model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options={"num_predict": 80, "temperature": 0.2})
    return r["message"]["content"].strip()

def main():
    threading.Thread(target=listen_unity, daemon=True).start()
    announced, spoken = None, []
    with sd.InputStream(device=DEVICE, samplerate=SR, channels=1,
                        dtype="float32", callback=callback):
        print("Waiting for Unity... (say 'exit program' or press Ctrl+C to stop)")
        buf, n = [], 0
        while True:
            if state["scenario"] and state["scenario"] != announced:
                announced = state["scenario"]
                spoken = []
                intro = SCENARIOS[announced]["intro"]
                print("\nSCENARIO:", intro)
                send({"type": "scenario_started", "text": intro})
                speak(intro)
                buf, n = [], 0
            try:
                data = q.get(timeout=0.5)
            except queue.Empty:
                continue
            buf.append(data)
            n += len(data)
            if n < SR * CHUNK_SEC:
                continue
            audio = np.concatenate(buf)
            buf, n = [], 0
            segments, _ = model.transcribe(
                audio, language="en", beam_size=1,
                vad_filter=True, condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segments)

            if text:
                print("You:", text)
                if any(p in text.lower() for p in EXIT_PHRASES):
                    print("Exit command heard.")
                    return
                if state["scenario"]:
                    spoken.append(text)
                    send({"type": "transcript", "text": text})
            elif spoken and state["scenario"]:       # worker stopped talking
                full = " ".join(spoken)
                spoken = []
                score, done, missed = evaluate(state["scenario"], full)
                send({"type": "feedback", "score": score,
                      "done": done, "missed": missed})
                print(f"Score: {score}%  Missed: {missed}")
                if missed:
                    speak(f"Your score is {score} percent. "
                          f"You missed: {'. '.join(missed)}.")
                    tip = coach(full, missed)
                    print("Coach:", tip)
                    send({"type": "coach", "text": tip})
                    speak(tip)
                else:
                    speak("Excellent. You completed every safety step.")
                state["scenario"], announced = None, None
                buf, n = [], 0

try:
    main()
except KeyboardInterrupt:
    print("\nStopping safely...")
finally:
    print("Program stopped. Goodbye.")