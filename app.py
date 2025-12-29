
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic
from translator_engine import GlossTranslator

# Page Config
st.set_page_config(layout="wide", page_title="OmniSign ISL", page_icon="🤟")

# Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-title {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">OmniSign ISL</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bi-directional Sign Language Translation System</div>', unsafe_allow_html=True)

# Initialize Translator
if 'translator' not in st.session_state:
    st.session_state.translator = GlossTranslator()

# Sidebar
st.sidebar.title("Settings")
mode = st.sidebar.radio("Mode", ["Sign to Text", "Text to Sign"])
target_lang_code = st.sidebar.selectbox(
    "Target Language", 
    options=list(st.session_state.translator.get_supported_languages().keys()),
    format_func=lambda x: st.session_state.translator.get_supported_languages()[x],
    index=0 # Default English
)

# Main App Logic
if mode == "Sign to Text":
    st.subheader("Live Sign Recognition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st_frame = st.empty()
        
    with col2:
        st.markdown("### Detected Output")
        st_gloss = st.empty()
        st_sentence = st.empty()
        st.markdown("---")
        st.info("Perform signs in front of the camera. Ensure good lighting.")

    # Camera Control
    run_camera = st.checkbox("Start Camera", value=False)
    
    if run_camera:
        status_text = st.empty()
        status_text.text("Initializing Camera...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera settings.")
        
        # Load MediaPipe Holistic
        elif mp_holistic:
            status_text.text("Loading AI Model...")
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                status_text.text("Running...")
                
                while cap.isOpened():
                    # Check if user stopped
                    # Streamlit handles this by interrupting, but explicit breaks help sometimes
                    if not run_camera:
                        break
                        
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from camera.")
                        break
                    
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # Logic for Prediction (Placeholder for Day 1)
                    # Simulating detection for demonstration
                    current_gloss = "WAITING..."
                    if results and (results.left_hand_landmarks or results.right_hand_landmarks):
                        # Mock detection logic
                        import random
                        if random.random() > 0.95: 
                           current_gloss = random.choice(["HELLO", "NAMASTE", "ME FINE", "HELP NEED"])
                    
                    # Translation
                    translated_text = st.session_state.translator.gloss_to_sentence(current_gloss, target_lang_code)
                    
                    # Update UI
                    st_frame.image(image, channels="BGR", use_column_width=True)
                    st_gloss.info(f"**Gloss:** `{current_gloss}`")
                    st_sentence.success(f"**Translated:** {translated_text}")
                    
            status_text.text("Stopped.")
            cap.release()
            
        else:
            st.error("MediaPipe Holistic functionality is not available. Please check installation.")
            cap.release()

elif mode == "Text to Sign":
    st.subheader("Text to ISL Animation")
    
    text_input = st.text_input("Enter text to translate (supports Hindi/English):", "Namaste")
    
    if st.button("Translate"):
        # 1. Convert Sentence -> Gloss
        gloss = st.session_state.translator.sentence_to_gloss(text_input, source_lang=target_lang_code) # Assuming input is in target lang for now, or detect
        
        st.success(f"Generated Gloss: {gloss}")
        
        st.info("Playing Animation Sequence (Placeholder)...")
        # Placeholder for .npy animation playback
        # In real app: loop through frames of the animation
        
        st.warning(f"Animation for '{gloss}' would play here.")
