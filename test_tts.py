import sys, subprocess, wave
import sounddevice as sd
import numpy as np

def speak(text):
    subprocess.run(
        [sys.executable, "-m", "piper", "-m", "en_US-lessac-medium",
         "--data-dir", "voices", "-f", "out.wav"],
        input=text.encode("utf-8"), check=True
    )
    with wave.open("out.wav", "rb") as w:
        sr = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    sd.play(audio, sr)
    sd.wait()

speak("Fire detected. Move to the nearest fire exit. Do not use the lift.")