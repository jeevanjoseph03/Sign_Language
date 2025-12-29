"""
text_to_sign_realtime.py
Integrates speech-to-text (Vosk) with text-to-sign animation for real-time accessibility.
"""
from speech_to_text import SpeechToText
from bi_directional_demo import BidirectionalDemoGUI
import tkinter as tk

# Initialize speech-to-text
stt = SpeechToText()

# Listen for speech and get text
print("Speak now. Listening for 5 seconds...")
spoken_text = stt.listen(duration=5)
print(f"Recognized: {spoken_text}")

# Launch GUI and display sign for recognized text
if spoken_text:
    root = tk.Tk()
    app = BidirectionalDemoGUI(root)
    # Insert recognized text into the text-to-sign input
    try:
        app.text_input.insert(0, spoken_text)
        app._on_display_click()
        root.mainloop()
    except Exception as e:
        print(f"Error displaying sign: {e}")
else:
    print("No speech recognized.")
