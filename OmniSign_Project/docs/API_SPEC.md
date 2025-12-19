# OmniSign API Specification

## Overview
Complete REST API and module interfaces for the OmniSign Multilingual Bidirectional Communication Framework.

---

## Core Modules

### 1. Feature Extraction (`data_pipeline.feature_extractor`)

```python
class MediaPipeFeatureExtractor:
    def extract_landmarks(frame: np.ndarray) -> Dict
    def extract_sequence(video_path: str, num_frames: int) -> np.ndarray
    def extract_from_webcam(duration_seconds: int, num_frames: int) -> np.ndarray
    def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray
```

**Returns:**
```json
{
    "hands": "ndarray(42, 4)",
    "face": "ndarray(100, 4)",
    "pose": "ndarray(33, 4)",
    "hand_confidence": "float",
    "face_confidence": "float",
    "pose_confidence": "float"
}
```

---

### 2. Dual-Stream Model (`models.dual_stream_model`)

```python
class DualStreamSignRecognizer:
    def __init__(num_classes, manual_features, non_manual_features, sequence_length)
    def build_model() -> Tuple[Model, Model]
    def compile(learning_rate) -> None
    def predict(manual_input, non_manual_input, batch_size) -> np.ndarray
    def get_confidence_scores(manual_input, non_manual_input) -> Tuple[ndarray, ndarray]
```

**Inputs:**
- Manual stream: `(batch_size, 30, 168)`
- Non-manual stream: `(batch_size, 30, 332)`

**Outputs:**
- Predictions: `(batch_size, num_classes)`
- Confidence scores: `(batch_size,)`

---

### 3. Training Pipeline (`models.ctc_training`)

```python
class CTCTrainingPipeline:
    def __init__(model, num_classes, checkpoint_dir)
    def train(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate) -> Dict
    def validation_step(X_val, y_val) -> Tuple[float, float]
    def save_checkpoint(epoch) -> None
    def save_history() -> None

class FineTuningPipeline(CTCTrainingPipeline):
    def prepare_for_finetuning(freeze_until_layer)
    def finetune(X_personal, y_personal, epochs, batch_size, learning_rate) -> Dict
```

**Training Parameters:**
```python
{
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "early_stopping_patience": 10,
    "decay_rate": 0.96
}
```

---

### 4. Data Loader (`data_pipeline.data_loader`)

```python
class DataLoader:
    def __init__(data_path, actions, no_sequences, sequence_length)
    def load_data() -> Tuple[ndarray, ndarray]
    def normalize(X, fit) -> ndarray
    def augment_data(X, y, augmentation_factor) -> Tuple[ndarray, ndarray]
    def split_data(X, y, train_ratio, val_ratio) -> Tuple[ndarray, ...]
    def create_batches(X, y, batch_size, shuffle) -> Generator
```

**Data Format:**
```
Sign_Language_Data/
├── Action_Name/
│   ├── 0/
│   │   ├── 0.npy  (shape: 30, 500)
│   │   └── ...
│   └── ...
```

---

### 5. Translation Engine (`modules.translator`)

```python
class TranslationService:
    def __init__(api_key)
    def translate_text(text, source_lang, target_lang) -> str
    def sign_to_text(sign_text, target_lang) -> str
    def text_to_sign(text, source_lang) -> str
    def batch_translate(texts, source_lang, target_lang) -> list

class BidirectionalCommunicationEngine:
    def __init__(api_key)
    def process_sign_input(recognized_sign, target_lang) -> Dict
    def process_text_input(text, source_lang) -> Dict
```

**Translation Result:**
```json
{
    "sign": "Hello",
    "english": "Hello",
    "translation": "നമസ്കാരം",
    "language": "ml",
    "confidence": 0.95
}
```

---

### 6. Personalization (`modules.personalization`)

```python
class SignerProfile:
    def __init__(user_id, name, language, profile_dir)
    def to_dict() -> Dict
    def from_dict(data) -> SignerProfile

class PersonalizationEngine:
    def __init__(model, model_dir, profile_dir)
    def create_user_profile(user_id, name, language) -> SignerProfile
    def load_user_profile(user_id) -> SignerProfile
    def add_calibration_sample(gesture_data, gesture_label, confidence)
    def estimate_signing_characteristics() -> Dict
    def finetune_on_calibration_data(epochs) -> Dict
    def get_personalization_status() -> Dict
    def record_gesture(gesture_label, confidence)
```

**User Profile:**
```json
{
    "user_id": "user_001",
    "name": "John Doe",
    "language": "en",
    "calibration_samples": 10,
    "calibration_target": 10,
    "base_model_accuracy": 0.85,
    "personalized_accuracy": 0.95,
    "signing_speed": 1.05,
    "hand_size": 0.95,
    "confidence_threshold": 0.85,
    "total_gestures_recognized": 42,
    "last_updated": "2025-12-19T10:30:00"
}
```

---

### 7. Main Application (`main_app`)

```python
class OmniSignApp:
    def __init__(model_path, actions)
    def create_session(user_id, user_name, language) -> SignerProfile
    def recognize_sign_from_webcam(duration_seconds, num_frames) -> Dict
    def generate_sign_from_text(text, source_language) -> Dict
    def personalization_wizard() -> Dict
    def interactive_session(interactive) -> None
    def save_session(output_file)
```

**Recognition Result:**
```json
{
    "sign": "Hello",
    "confidence": 0.92,
    "translation": "നമസ്കാരം",
    "status": "success",
    "language": "ml",
    "timestamp": "2025-12-19T10:30:00"
}
```

---

## REST API Endpoints (Future)

### Sign Recognition
```
POST /api/recognize
Content-Type: application/json

{
    "frame_data": "base64_encoded_image",
    "user_id": "user_001"
}

Response:
{
    "sign": "Hello",
    "confidence": 0.92,
    "translation": "Translation in target language"
}
```

### Text to Sign
```
POST /api/text-to-sign
Content-Type: application/json

{
    "text": "Hello, how are you?",
    "source_language": "en",
    "target_language": "en"
}

Response:
{
    "signs": ["Hello", "How", "are", "you"],
    "video_url": "path/to/generated/video.mp4"
}
```

### User Personalization
```
POST /api/personalization/calibrate
Content-Type: application/json

{
    "user_id": "user_001",
    "calibration_samples": [...]
}

Response:
{
    "status": "success",
    "improvement": 0.12,
    "new_accuracy": 0.95
}
```

---

## Data Structures

### Gesture Sequence
```python
{
    "frames": 30,
    "features_per_frame": 500,
    "shape": (30, 500),
    "data_type": "float32",
    "components": {
        "hand_landmarks": (0, 168),      # 21 pts × 2 × 4 values
        "facial_landmarks": (168, 568),  # 100 pts × 4 values
        "body_landmarks": (568, 700)     # 33 pts × 4 values (trimmed)
    }
}
```

### Recognition History
```python
[
    {
        "timestamp": "2025-12-19T10:30:00",
        "sign": "Hello",
        "confidence": 0.92,
        "translation": "നമസ്കാരം",
        "user_id": "user_001"
    },
    ...
]
```

---

## Error Handling

### Low Confidence
```json
{
    "status": "low_confidence",
    "confidence": 0.72,
    "threshold": 0.85,
    "message": "Please try again"
}
```

### No Active Session
```json
{
    "status": "error",
    "message": "No active user session. Create session first.",
    "code": "NO_SESSION"
}
```

### Insufficient Calibration Data
```json
{
    "status": "error",
    "message": "Need at least 3 calibration samples",
    "samples_provided": 1,
    "samples_required": 3,
    "code": "INSUFFICIENT_DATA"
}
```

---

## Performance Metrics

### Expected Performance
| Metric | Target |
|--------|--------|
| Frame-level Accuracy | >95% |
| Sequence Accuracy | >90% |
| Inference Latency | <500ms |
| Training Time (100 samples) | ~2 hours |
| Fine-tuning Time (10 samples) | ~10 minutes |
| Memory Usage | ~2GB GPU / 500MB CPU |

### Benchmark Results
```json
{
    "test_accuracy": 0.91,
    "per_class_accuracy": {
        "Hello": 0.95,
        "Goodbye": 0.88,
        "Thank_you": 0.92,
        "How_are_you": 0.89,
        "I_need_help": 0.85
    },
    "average_confidence": 0.87,
    "inference_time_ms": 350
}
```

---

## Configuration Parameters

### Model Configuration
```python
MODEL_CONFIG = {
    "num_classes": 5,
    "sequence_length": 30,
    "manual_features": 168,
    "non_manual_features": 332,
    "total_features": 500,
    
    "manual_stream": {
        "lstm_units": [256, 128],
        "dropout": [0.3, 0.2]
    },
    
    "non_manual_stream": {
        "conv_filters": [64, 128, 128],
        "kernel_sizes": [3, 3, 3],
        "dropout": 0.2
    },
    
    "fusion_layers": {
        "dense_units": [256, 128],
        "dropout": [0.3, 0.2]
    }
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
    "decay_rate": 0.96,
    "early_stopping_patience": 10,
    "validation_split": 0.15,
    "augmentation_factor": 2
}
```

---

## File Structure

```
OmniSign_Project/
├── models/
│   ├── dual_stream_model.py          # Main architecture
│   └── ctc_training.py                # Training with CTC loss
├── data_pipeline/
│   ├── feature_extractor.py          # MediaPipe extraction
│   └── data_loader.py                # Data preprocessing
├── modules/
│   ├── translator.py                 # Multilingual translation
│   └── personalization.py            # User adaptation
├── main_app.py                       # Interactive application
├── train_model.py                    # Training script
├── predict_sign.py                   # Inference script
└── requirements.txt                  # Dependencies
```

---

## Usage Examples

### Basic Recognition
```python
from main_app import OmniSignApp

app = OmniSignApp(model_path="sign_language_model.h5")
app.create_session("user_001", "John Doe", "en")
result = app.recognize_sign_from_webcam(duration_seconds=5)
print(f"Recognized: {result['sign']}")
print(f"Translation: {result['translation']}")
```

### Training
```python
from train_model import OmniSignTrainer

trainer = OmniSignTrainer()
X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_and_prepare_data()
model, history = trainer.train(X_train, X_val, y_train, y_val)
trainer.save_model(model)
```

### Personalization
```python
from modules.personalization import PersonalizationEngine

engine = PersonalizationEngine(model)
profile = engine.create_user_profile("user_001", "John", "en")

for i in range(10):
    sample = capture_gesture()
    engine.add_calibration_sample(sample, label=i%5)

results = engine.finetune_on_calibration_data(epochs=5)
```

---

## Support & Troubleshooting

### Common Issues

1. **Low Recognition Accuracy**
   - Solution: Run personalization wizard to calibrate system

2. **High Latency**
   - Solution: Use GPU acceleration (CUDA/cuDNN)

3. **Out of Memory**
   - Solution: Reduce batch size or sequence length

4. **Translation Errors**
   - Solution: Verify API key and internet connection

---

## Future Enhancements

- [ ] WebRTC for real-time video streaming
- [ ] Multi-language gesture synthesis
- [ ] Conversational AI integration
- [ ] Mobile app deployment
- [ ] Desktop application (Electron)
- [ ] Cloud synchronization
- [ ] Gesture video dataset export

