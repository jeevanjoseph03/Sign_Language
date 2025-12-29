
import random
from typing import List, Optional, Dict
from translation_utils import TranslationUtils

class GlossTranslator:
    """
    NLP Engine for converting ISL Glosses to natural sentences and vice-versa.
    """
    
    def __init__(self):
        self.translator = TranslationUtils(use_google_api=False)
        
        # Placeholder for Seq2Seq Model
        # In a real scenario, load: self.model = load_model('isl_gloss_to_text.h5')
        
        # Simple rule-based grammar correction for demo purposes
        self.grammar_rules = {
            "ME STORE GO": "I am going to the store",
            "NAME YOU WHAT": "What is your name?",
            "ME FINE": "I am fine",
            "YOU HOW": "How are you?",
            "HELP NEED": "I need help",
            "NAMASTE": "Namaste"
        }
        
    def gloss_to_sentence(self, gloss: str, target_lang: str = "en") -> str:
        """
        Converts Gloss (e.g., "ME STORE GO") to a proper sentence in target language.
        """
        gloss = gloss.upper().strip()
        
        # 1. Gloss -> English Sentence (Model inference or Rule-based)
        english_sentence = self.grammar_rules.get(gloss)
        
        if not english_sentence:
            # Fallback for unknown glosses: Just Capitalize and hope
            english_sentence = gloss.capitalize()
            # If gloss contains spaces, try to form a simple sentence
            words = gloss.split()
            if len(words) > 1:
                if words[0] == "ME":
                    words[0] = "I"
                english_sentence = " ".join(words).capitalize()

        # 2. English Sentence -> Target Language
        if target_lang == "en":
            return english_sentence
            
        return self.translator.translate_sign_to_text(english_sentence, target_lang)

    def sentence_to_gloss(self, text: str, source_lang: str = "en") -> str:
        """
        Converts a natural sentence to ISL Gloss.
        
        Logic:
        1. Translate Input -> English
        2. English NLP -> Extract Key Concepts -> Gloss
        """
        # 1. Translate to English if needed
        if source_lang != "en":
            # Reverse translation is tricky without an API that supports it explicitly,
            # but our translate_sign_to_text can work if we treat input as sign text
            # OR we just use the underlying googletrans
            text_en = self.translator._api_translate(text, source_lang, "en")
        else:
            text_en = text
            
        text_en_lower = text_en.lower().strip().replace("?", "").replace(".", "")
        
        # 2. Map to Gloss (Reverse lookup of our simple rules)
        for gloss, sent in self.grammar_rules.items():
            if sent.lower().replace("?", "").replace(".", "") == text_en_lower:
                return gloss
                
        # Fallback: Simple keyword extraction
        # Remove stopwords (is, am, are, the, to...)
        stopwords = {"is", "am", "are", "the", "to", "a", "an", "of", "in"}
        keywords = [w.upper() for w in text_en_lower.split() if w not in stopwords]
        return " ".join(keywords)

    def get_supported_languages(self):
        return self.translator.get_supported_languages()
