# src/data_preprocessing.py
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import unicodedata

class TextPreprocessor:
    def __init__(self):
        """
        Class for preprocessing text data for Indian languages
        """
        # Common stopwords in Indian languages
        self.stopwords = {
            'hindi': ['और', 'से', 'का', 'की', 'में', 'को', 'है', 'हो', 'था', 'थी'],
            'english': ['the', 'and', 'is', 'in', 'to', 'of', 'a'],
        }
    
    def clean_text(self, text):
        """
        Clean and normalize text for Indian languages
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        if not text:
            return ""
        
        # Normalize Unicode (important for Indian languages)
        text = unicodedata.normalize('NFC', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep Indian language characters
        # Keep Devanagari, Bengali, Tamil, Telugu, etc. scripts
        text = re.sub(r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove digits (optional)
        text = re.sub(r'\d+', '', text)
        
        return text.strip().lower()
    
    def encode_labels(self, labels):
        """
        Encode language labels to numerical values
        """
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        return encoded_labels, le
    
    def remove_stopwords(self, text, language='hindi'):
        """
        Remove stopwords (basic implementation)
        """
        words = text.split()
        if language in self.stopwords:
            filtered_words = [word for word in words if word not in self.stopwords[language]]
            return ' '.join(filtered_words)
        return text
    
    def get_text_statistics(self, texts):
        """
        Get basic statistics about the text data
        """
        if not texts:
            return {
                'avg_text_length': 0,
                'min_text_length': 0,
                'max_text_length': 0,
                'unique_characters': 0
            }
        
        lengths = [len(text) for text in texts if text]
        
        stats = {
            'avg_text_length': np.mean(lengths) if lengths else 0,
            'min_text_length': min(lengths) if lengths else 0,
            'max_text_length': max(lengths) if lengths else 0,
            'unique_characters': len(set(''.join(texts))) if texts else 0,
        }
        return stats