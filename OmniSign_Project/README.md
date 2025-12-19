# OmniSign: Multilingual Bidirectional Communication Framework

## 🌍 Overview

OmniSign is a cutting-edge AI system enabling **real-time, two-way communication** between sign language users and speakers of other languages. Unlike traditional systems that recognize isolated hand gestures, OmniSign uses advanced deep learning to:

- **Understand continuous sign sequences** (not just individual signs)
- **Capture contextual meaning** through facial expressions and body posture
- **Adapt to individual signing styles** through personalized learning
- **Support multiple languages** via a pivot-language architecture

## 🎯 Key Features

### 1. **Dual-Stream Intelligence**
- **Manual Stream:** LSTM networks analyzing hand movements and shapes
- **Non-Manual Stream:** CNN networks processing facial expressions and body posture
- Real-time fusion for context-aware recognition

### 2. **Bidirectional Communication**
- **Sign → Text/Speech:** Recognizes gestures and translates to spoken/written language
- **Speech/Text → Sign:** Converts user input to sign language for the deaf/hard-of-hearing

### 3. **Multilingual Support**
- English pivot-language architecture
- Google Cloud Translation API for 100+ languages
- Real-time translation without language-specific models

### 4. **Signer-Adaptive Learning**
- Personalized fine-tuning for individual users
- Learns unique signing styles over time
- Confidence-based adaptive threshold

### 5. **Ethical AI**
- Confidence scoring for transparent predictions
- Local data storage (privacy-first)
- Low-latency processing for accessibility

## 🏗️ Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed technical specifications.

```
Signer Gestures → MediaPipe Extraction → Dual-Stream Model → CTC Decoding → Translation
                                                                                    ↓
                                                    User's Native Language (Text/Audio)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Webcam (for real-time capture)

### Installation

```bash
# Clone/navigate to project
cd OmniSign_Project

# Create virtual environment
python -m venv sign_env
source sign_env/bin/activate  # On Windows: sign_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python check_env.py
```

### Data Collection
```bash
# Collect sign language samples
python collect_data.py

# Verify collected data
python verify_data.py
```

### Training
```bash
# Train the dual-stream model
python train_model.py

# Monitor training with TensorBoard
tensorboard --logdir=logs/
```

### Inference
```bash
# Real-time sign recognition and translation
python main_app.py
```

## 📊 Data Format

### Collection Structure
```
Sign_Language_Data/
├── Hello/
│   ├── 0/
│   │   ├── 0.npy  (30 frames × 500 features)
│   │   ├── 1.npy
│   │   └── ...
│   ├── 1/
│   └── ... (30 sequences total)
├── Goodbye/
├── Thank_you/
├── How_are_you/
└── I_need_help/
```

### Feature Vector (500 dims)
- Hand landmarks: 21 pts × 2 hands × 4 values (x,y,z,conf) = 168 dims
- Facial landmarks: ~100 × 4 = 400 dims
- Body landmarks: 33 × 4 = 132 dims
- **Total:** ~500 features per frame

## 🧠 Model Performance

| Metric | Target |
|--------|--------|
| Frame-level Accuracy | >95% |
| Sequence Accuracy | >90% |
| Latency | <500ms |
| Personalization Boost | +15-25% |
| Languages Supported | 100+ (via API) |

## 📚 Documentation

- [Architecture & Design](docs/ARCHITECTURE.md)
- [API Specification](docs/API_SPEC.md)
- [User Guide](docs/USER_GUIDE.md)

## 🔧 Project Structure

```
OmniSign_Project/
├── models/                    # Neural network architectures
├── data_pipeline/            # Data collection & preprocessing
├── modules/                  # Translation, personalization, etc.
├── ui/                       # User interface
├── docs/                     # Documentation
├── train_model.py           # Training script
├── predict_sign.py          # Inference script
├── main_app.py              # Application entry point
└── requirements.txt         # Dependencies
```

## 🔐 Privacy & Ethics

✅ **Local data storage** - No cloud transmission of personal data  
✅ **Confidence scoring** - Transparent predictions  
✅ **Personalization** - User-specific adaptation  
✅ **Accessibility** - Real-time processing for low-latency communication  

## 🎓 Use Cases

1. **Healthcare:** Seamless doctor-patient communication
2. **Education:** Real-time classroom interpretation
3. **Emergencies:** Crisis hotline interpretation
4. **Social Services:** Government/legal proceedings
5. **Technology:** Real-time video calls with interpretation

## 📞 Support

For issues, questions, or contributions, please contact the development team.

## 📄 License

[Specify your license here]

---

**OmniSign: Breaking barriers, enabling communication.**
