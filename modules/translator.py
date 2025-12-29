"""
Multilingual Translation Module

Provides bidirectional translation using a pivot-language architecture:
- Sign Language ← (English) → Spoken/Written Language
- Supports 100+ languages via Google Cloud Translation API
"""

import os
import json
from typing import Dict, Tuple, Optional
from enum import Enum


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    MALAYALAM = "ml"
    HINDI = "hi"
    TAMIL = "ta"
    TELUGU = "te"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    ARABIC = "ar"
    PORTUGUESE = "pt"


class TranslationService:
    """
    Handle multilingual translation using Google Cloud Translation API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize translation service.
        
        Args:
            api_key (str, optional): Google Cloud API key
        """
        self.api_key = api_key or os.getenv("GOOGLE_CLOUD_API_KEY")
        self.pivot_language = Language.ENGLISH.value  # English as pivot
        
        # Try to import Google Cloud client
        try:
            from google.cloud import translate_v2
            self.client = translate_v2.Client(api_key=self.api_key)
            self.api_available = True
        except ImportError:
            print("Warning: google-cloud-translate not installed")
            self.client = None
            self.api_available = False
        
        # Local translation cache
        self.translation_cache = {}
    
    def translate_text(self, text: str, source_language: str, 
                      target_language: str) -> str:
        """
        Translate text from source to target language.
        
        Uses pivot-language approach:
        - If source ≠ English: translate to English
        - Then translate English to target
        
        Args:
            text (str): Text to translate
            source_language (str): Source language code (e.g., 'ml', 'hi')
            target_language (str): Target language code
            
        Returns:
            str: Translated text
        """
        
        # Check cache
        cache_key = f"{text}:{source_language}→{target_language}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        translated_text = text
        
        if not self.api_available or self.client is None:
            print(f"Warning: Translation API not available. Returning original text.")
            return text
        
        try:
            # Step 1: Translate to English (pivot)
            if source_language != self.pivot_language:
                result = self.client.translate_text(
                    text,
                    source_language=source_language,
                    target_language=self.pivot_language
                )
                translated_text = result['translatedText']
            else:
                translated_text = text
            
            # Step 2: Translate from English to target
            if target_language != self.pivot_language:
                result = self.client.translate_text(
                    translated_text,
                    source_language=self.pivot_language,
                    target_language=target_language
                )
                translated_text = result['translatedText']
            
            # Cache result
            self.translation_cache[cache_key] = translated_text
            
        except Exception as e:
            print(f"Translation error: {e}")
            translated_text = text
        
        return translated_text
    
    def sign_to_text(self, sign_text: str, target_language: str = "en") -> str:
        """
        Convert recognized sign language text to spoken language.
        
        Args:
            sign_text (str): Recognized sign language text (in English)
            target_language (str): Target language code
            
        Returns:
            str: Translated text
        """
        
        if target_language == "en":
            return sign_text
        
        return self.translate_text(sign_text, "en", target_language)
    
    def text_to_sign(self, text: str, source_language: str = "en") -> str:
        """
        Convert spoken language text to sign language representation.
        
        Args:
            text (str): Text in source language
            source_language (str): Source language code
            
        Returns:
            str: Translated to English (pivot for sign language)
        """
        
        if source_language == "en":
            return text
        
        return self.translate_text(text, source_language, "en")
    
    def batch_translate(self, texts: list, source_language: str,
                       target_language: str) -> list:
        """
        Translate multiple texts efficiently.
        
        Args:
            texts (list): List of texts to translate
            source_language (str): Source language code
            target_language (str): Target language code
            
        Returns:
            list: Translated texts
        """
        
        translated = []
        for text in texts:
            translated.append(
                self.translate_text(text, source_language, target_language)
            )
        
        return translated


class SignLanguageVocabulary:
    """
    Manage sign language vocabulary and expressions.
    """
    
    # Sign language dictionary (sample)
    SIGN_VOCABULARY = {
        "Hello": {"type": "greeting", "duration": 1.5, "complexity": "low"},
        "Goodbye": {"type": "greeting", "duration": 1.0, "complexity": "low"},
        "Thank_you": {"type": "polite", "duration": 1.0, "complexity": "low"},
        "How_are_you": {"type": "question", "duration": 2.0, "complexity": "medium"},
        "I_need_help": {"type": "request", "duration": 2.5, "complexity": "high"},
        "Yes": {"type": "response", "duration": 0.5, "complexity": "low"},
        "No": {"type": "response", "duration": 0.5, "complexity": "low"},
        "Please": {"type": "polite", "duration": 0.8, "complexity": "low"},
        "Sorry": {"type": "polite", "duration": 1.0, "complexity": "low"},
    }
    
    # Multilingual dictionary for common phrases
    MULTILINGUAL_PHRASES = {
        "Hello": {
            "en": "Hello",
            "ml": "നമസ്കാരം",
            "hi": "नमस्ते",
            "es": "Hola",
            "fr": "Bonjour"
        },
        "Thank_you": {
            "en": "Thank you",
            "ml": "നന്ദി",
            "hi": "धन्यवाद",
            "es": "Gracias",
            "fr": "Merci"
        },
        "How_are_you": {
            "en": "How are you?",
            "ml": "നിങ്ങൾ എങ്ങനെയുണ്ട്?",
            "hi": "आप कैसे हैं?",
            "es": "¿Cómo estás?",
            "fr": "Comment allez-vous?"
        },
        "I_need_help": {
            "en": "I need help",
            "ml": "എനിക്ക് സഹായം വേണം",
            "hi": "मुझे मदद चाहिए",
            "es": "Necesito ayuda",
            "fr": "J'ai besoin d'aide"
        }
    }
    
    @classmethod
    def get_sign_info(cls, sign: str) -> Optional[Dict]:
        """Get information about a sign."""
        return cls.SIGN_VOCABULARY.get(sign)
    
    @classmethod
    def get_phrase_translation(cls, sign: str, language: str) -> str:
        """Get translation of a sign phrase."""
        if sign in cls.MULTILINGUAL_PHRASES:
            return cls.MULTILINGUAL_PHRASES[sign].get(language, "Unknown")
        return "Unknown phrase"
    
    @classmethod
    def list_signs(cls) -> list:
        """List all available signs."""
        return list(cls.SIGN_VOCABULARY.keys())


class BidirectionalCommunicationEngine:
    """
    Main engine for bidirectional communication between signs and language.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the communication engine.
        
        Args:
            api_key (str, optional): Google Cloud API key
        """
        self.translator = TranslationService(api_key)
        self.vocabulary = SignLanguageVocabulary()
        
    def process_sign_input(self, recognized_sign: str, 
                          target_language: str = "en") -> Dict:
        """
        Process recognized sign language input.
        
        Args:
            recognized_sign (str): Recognized sign (in English)
            target_language (str): Target language for output
            
        Returns:
            Dict with keys:
                - 'sign': Original recognized sign
                - 'english': English representation
                - 'translation': Translated text
                - 'language': Target language
                - 'confidence': Confidence level (1.0 if from vocabulary)
        """
        
        # Get phrase translation
        phrase_translation = self.vocabulary.get_phrase_translation(
            recognized_sign, target_language
        )
        
        # Use API for more natural translation if needed
        if target_language != "en":
            api_translation = self.translator.sign_to_text(
                recognized_sign, target_language
            )
        else:
            api_translation = recognized_sign
        
        return {
            "sign": recognized_sign,
            "english": recognized_sign,
            "translation": phrase_translation if phrase_translation != "Unknown phrase" else api_translation,
            "language": target_language,
            "confidence": 0.95  # High confidence for recognized signs
        }
    
    def process_text_input(self, text: str, source_language: str = "en") -> Dict:
        """
        Process text input in any language for sign generation.
        
        Args:
            text (str): Input text
            source_language (str): Source language code
            
        Returns:
            Dict with keys:
                - 'text': Original text
                - 'english': English translation (for sign generation)
                - 'source_language': Source language
                - 'signs_to_generate': List of signs to generate
        """
        
        # Translate to English
        english_text = self.translator.text_to_sign(text, source_language)
        
        # Parse into constituent signs
        signs = self._parse_text_to_signs(english_text)
        
        return {
            "text": text,
            "english": english_text,
            "source_language": source_language,
            "signs_to_generate": signs,
            "confidence": 0.90
        }
    
    def _parse_text_to_signs(self, text: str) -> list:
        """
        Parse English text into constituent signs.
        
        Args:
            text (str): English text
            
        Returns:
            list: List of signs to generate
        """
        
        # Simple keyword matching
        signs = []
        text_lower = text.lower()
        
        for sign in self.vocabulary.list_signs():
            if sign.lower().replace("_", " ") in text_lower:
                signs.append(sign)
        
        return signs if signs else ["Hello"]  # Default fallback


# Example usage
if __name__ == "__main__":
    print("Multilingual Translation Service Test")
    print("=" * 60)
    
    # Initialize engine
    engine = BidirectionalCommunicationEngine()
    
    # Test 1: Sign to text
    print("\n1. Sign to Text Translation:")
    print("-" * 40)
    result = engine.process_sign_input("Hello", target_language="ml")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Test 2: Text to sign
    print("\n2. Text to Sign Conversion:")
    print("-" * 40)
    result = engine.process_text_input("Thank you very much", source_language="en")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Test 3: Vocabulary
    print("\n3. Available Signs:")
    print("-" * 40)
    signs = engine.vocabulary.list_signs()
    for sign in signs:
        info = engine.vocabulary.get_sign_info(sign)
        print(f"  {sign}: {info['type']} (complexity: {info['complexity']})")
    
    # Test 4: Multilingual phrases
    print("\n4. Multilingual Phrases (Hello):")
    print("-" * 40)
    for lang in ["en", "ml", "hi", "es", "fr"]:
        phrase = engine.vocabulary.get_phrase_translation("Hello", lang)
        print(f"  {lang}: {phrase}")
