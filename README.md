Voice-Driven VR Safety Training for Construction Workers

Offline voice safety assistant: Whisper (speech to text), Ollama with Llama 3.2 1B
(guidance), and Piper (text to speech). Unity provides the virtual hazard scenes.

Files
- assistant.py: voice safety assistant (asks once, gives guidance, "ok" to stand by)
- training.py: training mode controlled by Unity over UDP
- live.py: basic live speech to text
- test_tts.py: Piper voice test

Setup
1. Install Python 3.11 and create a virtual environment
2. pip install -r requirements.txt
3. Install Ollama and run: ollama pull llama3.2:1b
4. Download a voice: python -m piper.download_voices en_US-lessac-medium --data-dir voices
5. Run: python assistant.py