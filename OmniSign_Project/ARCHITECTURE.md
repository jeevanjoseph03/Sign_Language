# OmniSign: Multilingual Bidirectional Communication Framework

## 🎯 Project Vision
A Multilingual, Bidirectional Communication Platform that enables real-time, two-way conversation between sign language users and speakers of other languages (e.g., Malayalam ↔ ASL). Unlike existing systems that only recognize isolated signs, OmniSign handles continuous sign sequences with contextual understanding.

---

## 🏗️ System Architecture

### Overview
```
┌─────────────────────────────────────────────────────────────┐
│            BIDIRECTIONAL COMMUNICATION FLOW                 │
└─────────────────────────────────────────────────────────────┘

SIGNER INPUT → MediaPipe Extraction → Dual-Stream Processing → Translation → Text/Audio Output
   ↑                                                                              ↓
   └──────────────────────── Signer-Adaptive Fine-Tuning ◄──────────────────────┘

SPEAKER INPUT → Speech/Text → Multilingual Translation (Pivot: English) → Sign Video Generation
```

### Core Components

#### 1. **Dual-Stream Architecture**

**Manual Stream (LSTM-based):**
- Processes hand landmarks (21 keypoints per hand × 2 hands = 42 inputs)
- Temporal modeling using Bidirectional LSTM
- Captures hand shape, position, and movement patterns
- Output: 256 hidden units

**Non-Manual Stream (CNN-based):**
- Facial expressions: 468 facial landmarks
- Body posture: 33 body keypoints
- Extracts spatial relationships and intensity
- Output: 128 features

**Fusion Layer:**
- Concatenates both streams (256 + 128 = 384 features)
- Dense layer with dropout for regularization
- Output: 256 features

**CTC Layer:**
- Handles continuous sequences without explicit frame-level alignment
- Allows variable-length input sequences
- Output: Softmax probabilities over action vocabulary

---

#### 2. **Pivot-Language Architecture**

```
Malayalam → English (Google Translate) → Sign Language (MediaPipe Generation)
   ↑                                              ↓
   └──────────── Signer Output (English) ◄──────┘
```

- **Central Hub:** English as the pivot language
- **Google Cloud APIs:** Used for real-time translation
- **Efficiency:** Reduces the need for direct language-to-language models

---

#### 3. **Signer-Adaptive Personalized Learning**

**Problem:** Every signer has unique hand shapes, speeds, and styles. Standard models fail to generalize.

**Solution:** Incremental fine-tuning
- Collect 5-10 personalized samples from each new user
- Fine-tune the last 2 Dense layers using low learning rate (1e-4)
- Update stored user profile with new patterns
- Confidence threshold: If model confidence < 0.85, ask for repeat + fine-tune

**User Profile Storage:**
```json
{
  "user_id": "user_001",
  "language": "Malayalam",
  "adapted_weights": {...},
  "calibration_samples": 15,
  "confidence_threshold": 0.85,
  "last_updated": "2025-12-19"
}
```

---

## 📊 Data Pipeline

### Data Collection
- **Framework:** MediaPipe Holistic
- **Per Frame:** 
  - Hand landmarks: 21 × 2 hands = 42 keypoints (x, y, z, confidence)
  - Facial landmarks: 468 keypoints (subset used: ~100)
  - Body landmarks: 33 keypoints
  - **Total:** ~500 features per frame

- **Sequence:** 30 frames per gesture (adjustable)
- **Vocabulary:** 5 actions initially (Hello, Goodbye, Thank you, How are you, I need help)

### Data Format
```
MP_Data/
├── Hello/
│   ├── 0/
│   │   ├── 0.npy (shape: (30, 500))  # Sequence of 30 frames, 500 features each
│   │   ├── 1.npy
│   │   └── ...
│   ├── 1/
│   └── ...
├── Goodbye/
└── ...
```

### Preprocessing
1. **Normalization:** Normalize landmarks to range [0, 1]
2. **Interpolation:** Handle missing frames using spline interpolation
3. **Augmentation:** 
   - Random flipping (left-right)
   - Random temporal shifts
   - Random scaling (±5%)
   - Gaussian noise (σ=0.01)

---

## 🧠 Model Architecture

### Layer-by-Layer Specification

```
Input Shape: (30, 500)  # 30 frames, 500 features per frame

MANUAL STREAM:
  ├─ Hand Features: (30, 168)  # 21 pts × 2 hands × 4 (x,y,z,conf)
  ├─ Bidirectional LSTM (256 units)
  │  ├─ Forward: 256 units
  │  └─ Backward: 256 units
  │  Output: (30, 512)
  └─ GlobalAveragePooling1D → (512,)

NON-MANUAL STREAM:
  ├─ Facial + Body Features: (30, 332)  # 100 facial + 33 body × 4
  ├─ Conv1D (filters=64, kernel=3) → ReLU
  ├─ MaxPooling1D(2)
  ├─ Conv1D (filters=128, kernel=3) → ReLU
  ├─ GlobalAveragePooling1D → (128,)
  └─ Dense(128) → ReLU

FUSION:
  ├─ Concatenate([512, 128]) → (640,)
  ├─ Dense(256) → ReLU → Dropout(0.3)
  ├─ Dense(128) → ReLU → Dropout(0.2)
  ├─ Dense(num_classes)
  └─ CTC Decoding (best path or beam search)

Loss: CTC Loss
Optimizer: Adam (lr=1e-3)
```

---

## 🔄 Workflow

### Phase 1: Data Collection & Preparation
1. Use MediaPipe Holistic to capture all landmarks
2. Store as .npy files with frame sequences
3. Verify data integrity and consistency

### Phase 2: Model Training
1. Load data in batches
2. Use CTC loss for continuous sequence modeling
3. Validate on held-out test set
4. Save best model weights

### Phase 3: Inference & Translation
1. **Sign → Text:** 
   - Capture signer's gestures
   - Extract landmarks using MediaPipe
   - Feed through dual-stream model
   - Get CTC decoded text
   - Translate to target language (using Google API)

2. **Text/Speech → Sign:**
   - Convert user speech to text (Google Speech-to-Text)
   - Translate to English (pivot language)
   - Generate synthetic sign video (gesture synthesis)
   - Display to signer

### Phase 4: Personalization
1. Collect 5-10 calibration samples from new user
2. Fine-tune last 2 layers with low learning rate
3. Store personalized weights
4. Use confidence threshold for continuous adaptation

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Pose Estimation** | MediaPipe Holistic |
| **Hand Landmark Detection** | MediaPipe Hands (21 pts/hand) |
| **Facial Recognition** | MediaPipe Face (468 pts) |
| **Sequence Modeling** | LSTM (Bidirectional) |
| **Spatial Feature Extraction** | CNN (Conv1D) |
| **Sequence-to-Sequence** | CTC Loss |
| **Multilingual Translation** | Google Cloud Translation API |
| **Speech-to-Text** | Google Cloud Speech-to-Text API |
| **UI Framework** | Tkinter/Gradio (Frontend) |
| **Backend** | Flask/FastAPI |
| **ML Framework** | TensorFlow/Keras |

---

## 📈 Expected Performance Metrics

- **Frame-level Accuracy:** >95% (hand landmarks)
- **Sequence Accuracy:** >90% (continuous gestures)
- **Latency:** <500ms per gesture
- **Multilingual Support:** 5+ languages (via pivot)
- **Personalization Improvement:** +15-25% accuracy after 10 samples

---

## 🚀 Development Timeline

**Day 1:** Architecture setup, dual-stream model implementation, data pipeline
**Day 2:** Training pipeline with CTC loss, basic inference
**Day 3:** Multilingual translation integration
**Day 4:** Signer-adaptive learning implementation
**Day 5:** UI and presentation

---

## 📝 Files Structure

```
OmniSign_Project/
├── models/
│   ├── dual_stream_model.py          # Main model architecture
│   ├── ctc_loss.py                   # CTC loss implementation
│   └── model_utils.py                # Model utilities
├── data_pipeline/
│   ├── data_loader.py                # Data loading and preprocessing
│   ├── feature_extractor.py          # MediaPipe landmark extraction
│   └── data_augmentation.py          # Data augmentation techniques
├── modules/
│   ├── translator.py                 # Multilingual translation module
│   ├── personalization.py            # Signer-adaptive learning
│   ├── gesture_synthesis.py          # Sign video generation
│   └── confidence_handler.py          # Confidence-based error handling
├── ui/
│   ├── app.py                        # Main application
│   └── components.py                 # UI components
├── docs/
│   ├── ARCHITECTURE.md               # This file
│   ├── API_SPEC.md                   # API specifications
│   └── USER_GUIDE.md                 # User manual
├── train_model.py                    # Training script
├── predict_sign.py                   # Inference script
├── main_app.py                       # Application entry point
├── requirements.txt                  # Dependencies
└── README.md                         # Project overview
```

---

## 🔐 Ethical AI Considerations

1. **Confidence Scoring:** Display confidence threshold; ask for repeat if <85%
2. **Data Privacy:** User personalization data stored locally, not transmitted
3. **Bias Mitigation:** Collect diverse signing styles during data collection
4. **Accessibility:** Low-latency processing for real-time communication
5. **Transparency:** Show which features the model used for prediction

