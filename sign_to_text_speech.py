"""
sign_to_text_speech.py
Recognizes sign language (via webcam), converts to text, and vocalizes using TTS.
"""
from bi_directional_demo import BidirectionalDemoGUI
import pyttsx3
import tkinter as tk

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Adjust speech rate for clarity

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Launch GUI for sign-to-text
root = tk.Tk()
app = BidirectionalDemoGUI(root)

# Assume app has a method or callback to get recognized text after sign input
# For demonstration, we simulate recognized text
recognized_text = "Hello, how are you?"  # Replace with actual recognition callback
print(f"Recognized text: {recognized_text}")
speak(recognized_text)

root.mainloop()
