"""
speech_to_text.py
Efficient real-time speech-to-text transcription using Vosk (offline, free).
"""
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json

class SpeechToText:
    def __init__(self, model_path="model", sample_rate=16000):
        try:
            self.model = Model(model_path)
            self.sample_rate = sample_rate
            self.q = queue.Queue()
            self.rec = KaldiRecognizer(self.model, self.sample_rate)
            self.stream = None
            self.available = True
        except Exception as e:
            print(f"[WARNING] Vosk model not found at {model_path}. Speech-to-text disabled. Error: {e}")
            self.available = False

    def _callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    def listen(self, duration=5):
        """Listen for a fixed duration and return transcribed text."""
        if not self.available:
            return "Speech-to-text not available. Please download Vosk model."
        self.q.queue.clear()
        with sd.RawInputStream(samplerate=self.sample_rate, blocksize = 8000, dtype='int16', channels=1, callback=self._callback):
            print(f"Listening for {duration} seconds...")
            sd.sleep(int(duration * 1000))
            result = ""
            while not self.q.empty():
                data = self.q.get()
                if self.rec.AcceptWaveform(data):
                    res = json.loads(self.rec.Result())
                    result += res.get('text', '') + " "
            # Final partial result
            res = json.loads(self.rec.FinalResult())
            result += res.get('text', '')
        return result.strip()

if __name__ == "__main__":
    stt = SpeechToText()
    print("You said:", stt.listen(5))
