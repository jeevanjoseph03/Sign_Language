"""
Multilingual Translation Utilities for OmniSign

Provides translation services for Sign Language recognition results
into multiple languages using Google Translate API and fallback methods.

Supported Languages:
- English (en)
- Spanish (es)
- French (fr)
- Arabic (ar)
- German (de)
- Portuguese (pt)
- Chinese Simplified (zh-CN)
- Japanese (ja)
"""

import os
from typing import Dict, List, Optional
from enum import Enum
import json
from pathlib import Path


class Language(Enum):
    """Supported languages for translation."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    ARABIC = "ar"
    GERMAN = "de"
    PORTUGUESE = "pt"
    CHINESE = "zh-CN"
    JAPANESE = "ja"


class TranslationUtils:
    """
    Utility class for translating sign language recognition results
    into multiple languages.
    
    Supports:
    1. Google Translate API (primary)
    2. googletrans library (fallback)
    3. Local translation dictionary (for common signs)
    """
    
    # Local translation dictionary for common sign language phrases
    LOCAL_TRANSLATIONS = {
        "hello": {
            "en": "Hello",
            "es": "Hola",
            "fr": "Bonjour",
            "ar": "مرحبا",
            "de": "Hallo",
            "pt": "Olá",
            "zh-CN": "你好",
            "ja": "こんにちは"
        },
        "goodbye": {
            "en": "Goodbye",
            "es": "Adiós",
            "fr": "Au revoir",
            "ar": "وداعا",
            "de": "Auf Wiedersehen",
            "pt": "Adeus",
            "zh-CN": "再见",
            "ja": "さようなら"
        },
        "thank you": {
            "en": "Thank you",
            "es": "Gracias",
            "fr": "Merci",
            "ar": "شكرا",
            "de": "Danke",
            "pt": "Obrigado",
            "zh-CN": "谢谢",
            "ja": "ありがとう"
        },
        "how are you": {
            "en": "How are you?",
            "es": "¿Cómo estás?",
            "fr": "Comment allez-vous?",
            "ar": "كيف حالك؟",
            "de": "Wie geht es dir?",
            "pt": "Como você está?",
            "zh-CN": "你好吗？",
            "ja": "お元気ですか？"
        },
        "i need help": {
            "en": "I need help",
            "es": "Necesito ayuda",
            "fr": "J'ai besoin d'aide",
            "ar": "أحتاج إلى مساعدة",
            "de": "Ich brauche Hilfe",
            "pt": "Preciso de ajuda",
            "zh-CN": "我需要帮助",
            "ja": "助けが必要です"
        },
        "yes": {
            "en": "Yes",
            "es": "Sí",
            "fr": "Oui",
            "ar": "نعم",
            "de": "Ja",
            "pt": "Sim",
            "zh-CN": "是",
            "ja": "はい"
        },
        "no": {
            "en": "No",
            "es": "No",
            "fr": "Non",
            "ar": "لا",
            "de": "Nein",
            "pt": "Não",
            "zh-CN": "否",
            "ja": "いいえ"
        },
        "please": {
            "en": "Please",
            "es": "Por favor",
            "fr": "S'il vous plaît",
            "ar": "من فضلك",
            "de": "Bitte",
            "pt": "Por favor",
            "zh-CN": "请",
            "ja": "お願いします"
        },
        "sorry": {
            "en": "Sorry",
            "es": "Lo siento",
            "fr": "Désolé",
            "ar": "آسف",
            "de": "Entschuldigung",
            "pt": "Desculpe",
            "zh-CN": "对不起",
            "ja": "すみません"
        },
        "okay": {
            "en": "Okay",
            "es": "Está bien",
            "fr": "D'accord",
            "ar": "حسنا",
            "de": "Okay",
            "pt": "Tudo bem",
            "zh-CN": "好的",
            "ja": "了解"
        }
    }
    
    def __init__(self, use_google_api: bool = False, api_key: Optional[str] = None):
        """
        Initialize translation utilities.
        
        Args:
            use_google_api (bool): Whether to use Google Cloud Translation API
            api_key (str, optional): Google Cloud API key (uses env var if not provided)
        """
        self.use_google_api = use_google_api
        self.api_key = api_key or os.getenv("GOOGLE_CLOUD_API_KEY")
        self.google_client = None
        self.googletrans_available = False
        
        # Initialize Google Cloud client if requested
        if self.use_google_api and self.api_key:
            try:
                from google.cloud import translate_v2
                self.google_client = translate_v2.Client(api_key=self.api_key)
                print("[OK] Google Cloud Translation API initialized")
            except ImportError:
                print("[WARNING] google-cloud-translate not available, falling back to googletrans")
        
        # Try to import googletrans as fallback
        try:
            from googletrans import Translator
            self.translator = Translator()
            self.googletrans_available = True
            print("[OK] googletrans library available")
        except ImportError:
            print("[WARNING] googletrans not installed, will use local translations only")
            self.translator = None
    
    def translate_sign_to_text(self, sign_label: str, 
                               target_language: str = "en") -> str:
        """
        Translate a recognized sign label into target language.
        
        Args:
            sign_label (str): Recognized sign label (e.g., "Hello")
            target_language (str): Target language code (default: "en")
            
        Returns:
            str: Translated text
        """
        # Normalize sign label
        normalized_sign = sign_label.lower().strip()
        
        # Try local dictionary first (most reliable)
        if normalized_sign in self.LOCAL_TRANSLATIONS:
            if target_language in self.LOCAL_TRANSLATIONS[normalized_sign]:
                return self.LOCAL_TRANSLATIONS[normalized_sign][target_language]
        
        # Fall back to API-based translation
        return self._api_translate(sign_label, "en", target_language)
    
    def translate_text_to_sign_lookup(self, text: str) -> Optional[str]:
        """
        Reverse lookup to find sign label from text input.
        
        Args:
            text (str): Input text (e.g., "Hola")
            
        Returns:
            str: Sign label if found (e.g., "hello"), None otherwise
        """
        text_normalized = text.lower().strip()
        
        # Search through translation dictionary
        for sign, translations in self.LOCAL_TRANSLATIONS.items():
            for lang_code, translated_text in translations.items():
                if translated_text.lower().strip() == text_normalized:
                    return sign
        
        return None
    
    def get_multilingual_output(self, sign_label: str, 
                                languages: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Get translation of sign label in multiple languages.
        
        Args:
            sign_label (str): Recognized sign label
            languages (list, optional): List of language codes. 
                                       Defaults to ['en', 'es', 'fr', 'ar']
            
        Returns:
            Dict[str, str]: Translations by language code
        """
        if languages is None:
            languages = ["en", "es", "fr", "ar"]
        
        result = {}
        for lang in languages:
            result[lang] = self.translate_sign_to_text(sign_label, lang)
        
        return result
    
    def _api_translate(self, text: str, source_lang: str, 
                      target_lang: str) -> str:
        """
        Use API to translate text.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            str: Translated text, or original if translation fails
        """
        # If target is same as source, return original
        if source_lang == target_lang:
            return text
        
        # Try Google Cloud API first
        if self.google_client:
            try:
                result = self.google_client.translate_text(
                    text,
                    source_language=source_lang,
                    target_language=target_lang
                )
                return result.get('translatedText', text)
            except Exception as e:
                print(f"[WARNING] Google Cloud API error: {e}")
        
        # Fall back to googletrans
        if self.googletrans_available and self.translator:
            try:
                result = self.translator.translate(text, src_lang=source_lang, dest_lang=target_lang)
                return result.get('text', text)
            except Exception as e:
                print(f"[WARNING] googletrans error: {e}")
        
        # If all else fails, return original text
        return text
    
    def save_translation_cache(self, filepath: str):
        """
        Save local translation dictionary to JSON file.
        
        Args:
            filepath (str): Path to save JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.LOCAL_TRANSLATIONS, f, ensure_ascii=False, indent=2)
        print(f"[OK] Translation cache saved to {filepath}")
    
    def load_translation_cache(self, filepath: str):
        """
        Load translation dictionary from JSON file.
        
        Args:
            filepath (str): Path to load JSON file from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.LOCAL_TRANSLATIONS.update(json.load(f))
        print(f"[OK] Translation cache loaded from {filepath}")
    
    @staticmethod
    def get_language_name(lang_code: str) -> str:
        """
        Get human-readable language name from code.
        
        Args:
            lang_code (str): Language code (e.g., 'es')
            
        Returns:
            str: Language name (e.g., 'Spanish')
        """
        language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "ar": "Arabic",
            "de": "German",
            "pt": "Portuguese",
            "zh-CN": "Chinese (Simplified)",
            "ja": "Japanese"
        }
        return language_names.get(lang_code, lang_code)
    
    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """
        Get all supported languages.
        
        Returns:
            Dict[str, str]: Language codes and names
        """
        return {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "ar": "Arabic",
            "de": "German",
            "pt": "Portuguese",
            "zh-CN": "Chinese (Simplified)",
            "ja": "Japanese"
        }
    
    def add_custom_translation(self, sign_label: str, 
                              language_translations: Dict[str, str]):
        """
        Add custom translation entry.
        
        Args:
            sign_label (str): Sign label (in lowercase)
            language_translations (dict): Dict with language codes and translations
        """
        normalized_sign = sign_label.lower().strip()
        self.LOCAL_TRANSLATIONS[normalized_sign] = language_translations
        print(f"[OK] Added translation for '{normalized_sign}'")


# Convenience functions for quick access
def translate_sign(sign_label: str, target_language: str = "en") -> str:
    """
    Quick function to translate a sign to target language.
    
    Args:
        sign_label (str): Recognized sign
        target_language (str): Target language code
        
    Returns:
        str: Translated text
    """
    utils = TranslationUtils()
    return utils.translate_sign_to_text(sign_label, target_language)


def get_multilingual_text(sign_label: str, 
                          languages: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Quick function to get multilingual translations.
    
    Args:
        sign_label (str): Recognized sign
        languages (list, optional): Target languages
        
    Returns:
        Dict[str, str]: Translations by language
    """
    utils = TranslationUtils()
    return utils.get_multilingual_output(sign_label, languages)


if __name__ == "__main__":
    # Example usage
    translator = TranslationUtils()
    
    # Test sign-to-text translation
    print("\n=== Sign-to-Text Translation ===")
    signs = ["Hello", "Goodbye", "Thank you", "How are you", "I need help"]
    
    for sign in signs:
        print(f"\n{sign}:")
        translations = translator.get_multilingual_output(sign)
        for lang, text in translations.items():
            print(f"  {TranslationUtils.get_language_name(lang)}: {text}")
    
    # Test text-to-sign lookup
    print("\n\n=== Text-to-Sign Lookup ===")
    texts = ["Hola", "Merci", "مرحبا", "再见"]
    for text in texts:
        sign = translator.translate_text_to_sign_lookup(text)
        print(f"'{text}' → Sign: {sign}")
