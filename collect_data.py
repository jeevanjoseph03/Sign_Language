"""
Data Collection Module for OmniSign Sign Language Recognition

Captures video sequences of sign language and extracts keypoints using MediaPipe.
Updated to support multilingual language mappings for sign labels.

Features:
- Real-time webcam capture with visual feedback
- Automatic keypoint extraction (including face landmarks for non-manual markers)
- Multilingual label mappings
- Sequence-based data organization
"""

import os
import time
from pathlib import Path
import json

import cv2
import numpy as np

from utils import (
    create_holistic,
    mediapipe_detection,
    extract_keypoints,
    draw_landmarks,
)


# Configuration - EXPANDED SIGN VOCABULARY
ACTIONS = [
    # Greetings (5)
    "Hello",
    "Goodbye",
    "Good morning",
    "Good evening",
    "Welcome",
    
    # Common Phrases (5)
    "Thank you",
    "Please",
    "Yes",
    "No",
    "Okay",
    
    # Questions (5)
    "How are you",
    "What is your name",
    "Where are you from",
    "Do you understand",
    "Can you help",
    
    # Needs & Emotions (5)
    "I need help",
    "I am happy",
    "I am sad",
    "I am tired",
    "I love you",
    
    # Common Actions (5)
    "Wait",
    "Stop",
    "Go",
    "Come here",
    "Sit down",
]

NO_SEQUENCES = 30  # Number of video sequences per sign
SEQUENCE_LENGTH = 30  # Number of frames per sequence
DATA_PATH = Path("Sign_Language_Data")

# Multilingual Label Mapping Dictionary
MULTILINGUAL_LABELS = {
    "Hello": {
        "en": "Hello",
        "es": "Hola",
        "fr": "Bonjour",
        "ar": "مرحبا",
        "de": "Hallo",
        "pt": "Olá",
        "zh-CN": "你好",
        "ja": "こんにちは"
    },
    "Goodbye": {
        "en": "Goodbye",
        "es": "Adiós",
        "fr": "Au revoir",
        "ar": "وداعا",
        "de": "Auf Wiedersehen",
        "pt": "Adeus",
        "zh-CN": "再见",
        "ja": "さようなら"
    },
    "Thank you": {
        "en": "Thank you",
        "es": "Gracias",
        "fr": "Merci",
        "ar": "شكرا",
        "de": "Danke",
        "pt": "Obrigado",
        "zh-CN": "谢谢",
        "ja": "ありがとう"
    },
    "How are you": {
        "en": "How are you?",
        "es": "¿Cómo estás?",
        "fr": "Comment allez-vous?",
        "ar": "كيف حالك؟",
        "de": "Wie geht es dir?",
        "pt": "Como você está?",
        "zh-CN": "你好吗？",
        "ja": "お元気ですか？"
    },
    "I need help": {
        "en": "I need help",
        "es": "Necesito ayuda",
        "fr": "J'ai besoin d'aide",
        "ar": "أحتاج إلى مساعدة",
        "de": "Ich brauche Hilfe",
        "pt": "Preciso de ajuda",
        "zh-CN": "我需要帮助",
        "ja": "助けが必要です"
    },
    "Good morning": {
        "en": "Good morning",
        "es": "Buenos días",
        "fr": "Bonjour",
        "ar": "صباح الخير",
        "de": "Guten Morgen",
        "pt": "Bom dia",
        "zh-CN": "早上好",
        "ja": "おはよう"
    },
    "Good evening": {
        "en": "Good evening",
        "es": "Buenas noches",
        "fr": "Bonsoir",
        "ar": "مساء الخير",
        "de": "Guten Abend",
        "pt": "Boa noite",
        "zh-CN": "晚上好",
        "ja": "こんばんは"
    },
    "Welcome": {
        "en": "Welcome",
        "es": "Bienvenido",
        "fr": "Bienvenue",
        "ar": "أهلا وسهلا",
        "de": "Willkommen",
        "pt": "Bem-vindo",
        "zh-CN": "欢迎",
        "ja": "ようこそ"
    },
    "Please": {
        "en": "Please",
        "es": "Por favor",
        "fr": "S'il vous plaît",
        "ar": "من فضلك",
        "de": "Bitte",
        "pt": "Por favor",
        "zh-CN": "请",
        "ja": "お願いします"
    },
    "Yes": {
        "en": "Yes",
        "es": "Sí",
        "fr": "Oui",
        "ar": "نعم",
        "de": "Ja",
        "pt": "Sim",
        "zh-CN": "是的",
        "ja": "はい"
    },
    "No": {
        "en": "No",
        "es": "No",
        "fr": "Non",
        "ar": "لا",
        "de": "Nein",
        "pt": "Não",
        "zh-CN": "不",
        "ja": "いいえ"
    },
    "Okay": {
        "en": "Okay",
        "es": "Está bien",
        "fr": "D'accord",
        "ar": "حسنا",
        "de": "Okay",
        "pt": "Tudo bem",
        "zh-CN": "好的",
        "ja": "わかりました"
    },
    "What is your name": {
        "en": "What is your name?",
        "es": "¿Cuál es tu nombre?",
        "fr": "Quel est votre nom?",
        "ar": "ما اسمك؟",
        "de": "Wie heißt du?",
        "pt": "Qual é seu nome?",
        "zh-CN": "你叫什么名字？",
        "ja": "お名前は？"
    },
    "Where are you from": {
        "en": "Where are you from?",
        "es": "¿De dónde eres?",
        "fr": "D'où venez-vous?",
        "ar": "من أين أنت؟",
        "de": "Woher kommst du?",
        "pt": "De onde você é?",
        "zh-CN": "你来自哪里？",
        "ja": "どこから来ましたか？"
    },
    "Do you understand": {
        "en": "Do you understand?",
        "es": "¿Entiendes?",
        "fr": "Comprenez-vous?",
        "ar": "هل تفهم؟",
        "de": "Verstehst du?",
        "pt": "Você entende?",
        "zh-CN": "你明白吗？",
        "ja": "わかりますか？"
    },
    "Can you help": {
        "en": "Can you help?",
        "es": "¿Puedes ayudar?",
        "fr": "Pouvez-vous aider?",
        "ar": "هل يمكنك المساعدة؟",
        "de": "Kannst du helfen?",
        "pt": "Você pode ajudar?",
        "zh-CN": "你能帮忙吗？",
        "ja": "手伝ってくれますか？"
    },
    "I am happy": {
        "en": "I am happy",
        "es": "Estoy feliz",
        "fr": "Je suis heureux",
        "ar": "أنا سعيد",
        "de": "Ich bin glücklich",
        "pt": "Estou feliz",
        "zh-CN": "我很高兴",
        "ja": "私は幸せです"
    },
    "I am sad": {
        "en": "I am sad",
        "es": "Estoy triste",
        "fr": "Je suis triste",
        "ar": "أنا حزين",
        "de": "Ich bin traurig",
        "pt": "Estou triste",
        "zh-CN": "我很伤心",
        "ja": "私は悲しいです"
    },
    "I am tired": {
        "en": "I am tired",
        "es": "Estoy cansado",
        "fr": "Je suis fatigué",
        "ar": "أنا متعب",
        "de": "Ich bin müde",
        "pt": "Estou cansado",
        "zh-CN": "我很累",
        "ja": "疲れています"
    },
    "I love you": {
        "en": "I love you",
        "es": "Te amo",
        "fr": "Je t'aime",
        "ar": "أنا أحبك",
        "de": "Ich liebe dich",
        "pt": "Eu te amo",
        "zh-CN": "我爱你",
        "ja": "愛しています"
    },
    "Wait": {
        "en": "Wait",
        "es": "Espera",
        "fr": "Attends",
        "ar": "انتظر",
        "de": "Warte",
        "pt": "Espere",
        "zh-CN": "等等",
        "ja": "待って"
    },
    "Stop": {
        "en": "Stop",
        "es": "Para",
        "fr": "Arrête",
        "ar": "قف",
        "de": "Stopp",
        "pt": "Pare",
        "zh-CN": "停止",
        "ja": "止まって"
    },
    "Go": {
        "en": "Go",
        "es": "Vete",
        "fr": "Allez",
        "ar": "اذهب",
        "de": "Geh",
        "pt": "Vá",
        "zh-CN": "去吧",
        "ja": "行って"
    },
    "Come here": {
        "en": "Come here",
        "es": "Ven aquí",
        "fr": "Viens ici",
        "ar": "تعال هنا",
        "de": "Komm her",
        "pt": "Venha aqui",
        "zh-CN": "过来",
        "ja": "ここに来て"
    },
    "Sit down": {
        "en": "Sit down",
        "es": "Siéntate",
        "fr": "Assieds-toi",
        "ar": "اجلس",
        "de": "Setz dich hin",
        "pt": "Sente-se",
        "zh-CN": "坐下",
        "ja": "座ってください"
    }
}


def ensure_directories():
    """Create required directories for data collection."""
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            seq_dir = DATA_PATH / action / str(sequence)
            os.makedirs(seq_dir, exist_ok=True)
    print(f"[OK] Created directories for {len(ACTIONS)} actions")


def save_multilingual_labels():
    """Save multilingual labels to JSON file."""
    labels_path = DATA_PATH / "labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(MULTILINGUAL_LABELS, f, ensure_ascii=False, indent=2)
    print(f"[OK] Multilingual labels saved to {labels_path}")


def load_multilingual_labels(filepath: str = None):
    """Load multilingual labels from JSON file."""
    if filepath is None:
        filepath = DATA_PATH / "labels.json"
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return MULTILINGUAL_LABELS


def main():
    """Enhanced data collection with progress tracking and quality feedback."""
    ensure_directories()
    save_multilingual_labels()
    
    print("=" * 60)
    print("OmniSign - Enhanced Data Collection System")
    print("=" * 60)
    print(f"Total signs to collect: {len(ACTIONS)}")
    print(f"Sequences per sign: {NO_SEQUENCES}")
    print(f"Frames per sequence: {SEQUENCE_LENGTH}")
    print(f"Total frames to collect: {len(ACTIONS) * NO_SEQUENCES * SEQUENCE_LENGTH}")
    print("=" * 60)
    print()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible: ensure camera permissions and device availability.")

    # Configure camera for optimal capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    collected_count = 0
    total_frames = len(ACTIONS) * NO_SEQUENCES * SEQUENCE_LENGTH
    
    try:
        with create_holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for action_idx, action in enumerate(ACTIONS):
                print(f"\n[{action_idx + 1}/{len(ACTIONS)}] Collecting: {action}")
                print("-" * 50)
                
                for sequence in range(NO_SEQUENCES):
                    sequence_keypoints = []
                    
                    for frame_num in range(SEQUENCE_LENGTH):
                        ret, frame = cap.read()
                        if not ret:
                            raise RuntimeError("Failed to read frame from webcam.")

                        image, results = mediapipe_detection(frame, holistic)
                        draw_landmarks(image, results)

                        # Create progress bar
                        progress = (collected_count / total_frames) * 100
                        bar_length = 30
                        filled = int(bar_length * collected_count / total_frames)
                        bar = "█" * filled + "░" * (bar_length - filled)
                        
                        # Display status
                        if frame_num == 0:
                            cv2.putText(
                                image,
                                "GET READY!",
                                (150, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 255, 0),
                                3,
                            )
                            cv2.putText(
                                image,
                                f"Starting in 2 seconds...",
                                (100, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 0),
                                2,
                            )
                            cv2.imshow("OmniSign - Data Collection", image)
                            cv2.waitKey(2000)
                        else:
                            # Display comprehensive info
                            cv2.putText(image, f"Sign: {action}", (20, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(image, f"Sequence: {sequence + 1}/{NO_SEQUENCES}", (20, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            cv2.putText(image, f"Frame: {frame_num}/{SEQUENCE_LENGTH}", (20, 90),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            cv2.putText(image, f"Progress: {progress:.1f}%", (20, 120),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            cv2.putText(image, f"[{bar}]", (20, 450),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
                            
                            cv2.imshow("OmniSign - Data Collection", image)
                            cv2.waitKey(30)

                        # Save keypoints
                        keypoints = extract_keypoints(results)
                        sequence_keypoints.append(keypoints)
                        save_path = DATA_PATH / action / str(sequence) / f"{frame_num}.npy"
                        np.save(save_path, keypoints)
                        
                        collected_count += 1

                        # Check for quit signal
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("\n[!] Quit signal received. Saving progress...")
                            cap.release()
                            cv2.destroyAllWindows()
                            print(f"[OK] Collected {collected_count}/{total_frames} frames")
                            return

                    # Data quality feedback
                    valid_frames = sum(1 for kp in sequence_keypoints if kp is not None)
                    quality = (valid_frames / SEQUENCE_LENGTH) * 100
                    status = "✓" if quality >= 80 else "⚠"
                    print(f"  Seq {sequence + 1:2d}/{NO_SEQUENCES}: {status} Quality {quality:.1f}%")

    except KeyboardInterrupt:
        print("\n[!] Collection interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print(f"[OK] Data collection complete!")
        print(f"[OK] Total frames collected: {collected_count}/{total_frames}")
        print(f"[OK] Signs collected: {len(ACTIONS)}")
        print("=" * 60)


if __name__ == "__main__":
    main()
