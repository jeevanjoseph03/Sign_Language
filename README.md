# OmniSign
 
## Overview

OmniSign is an advanced AI-powered platform for real-time, bidirectional communication between sign language users and speakers of any language. Designed for accessibility, OmniSign enables seamless translation between sign, text, and speech—empowering people who cannot speak or hear.
## Features

- Real-time sign language gesture recognition
- Speech-to-sign and sign-to-speech translation
- Multilingual support (100+ languages)
- Personalization for individual signing styles
- Privacy-first: offline speech recognition and local data storage
- Modern, user-friendly interface
- Modular codebase for easy extension

## Technologies Used

- Python
- OpenCV
- TensorFlow/Keras or PyTorch
- NumPy, Pandas, and other scientific libraries


## Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for sign recognition)
- Microphone (for speech input)
- Speakers/Headphones (for speech output)

### Installation

```bash
git clone https://github.com/jeevanjoseph03/OmniSign.git
cd OmniSign
python -m venv sign_env
sign_env\Scripts\activate  # On Windows
pip install -r requirements.txt
pip install vosk sounddevice pyttsx3 opencv-python
```

### Optional: Download Vosk Model
For offline speech recognition, download a Vosk model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and extract to the `model` folder.

### Run the Application
```bash
python bi_directional_demo.py
```
Choose your preferred mode:
- Speak to Sign
- Sign to Speech
- Text Chat

### Test All Features
```bash
python test_all_features.py
```

## Project Structure

```
OmniSign/
├── models/              # Neural network architectures
├── data_pipeline/       # Data collection & preprocessing
├── modules/             # Translation, personalization, etc.
├── docs/                # Documentation
├── train_model.py       # Training script
├── bi_directional_demo.py # Main GUI application
├── requirements.txt     # Dependencies
└── ...                  # Other scripts and resources
```
## Documentation

- [Architecture & Design](docs/ARCHITECTURE.md)
- [API Specification](docs/API_SPEC.md)
- [User Guide](docs/USER_GUIDE.md)

## Contributing

Contributions are welcome! Please fork the repository and open a pull request.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgements

Thanks to the open-source community and all contributors to sign language technology.

## Contact

For support, open an issue or contact [jeevanjoseph03](https://github.com/jeevanjoseph03).
     