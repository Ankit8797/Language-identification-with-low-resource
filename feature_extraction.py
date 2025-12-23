# src/feature_extraction.py - ENHANCED FOR LANGUAGE-SPECIFIC FEATURES
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack, csr_matrix
import re
from collections import Counter
import unicodedata

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        
        # Language-specific character sets
        self.language_markers = {
            'Hindi': ['ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', 'ऋ', 'ॠ', 'ऌ', 'ॡ'],
            'Marathi': ['ळ', 'श्र', 'क्ष', 'ज्ञ', 'त्र'],
            'Bengali': ['া', 'ি', 'ী', 'ু', 'ূ', 'ে', 'ৈ', 'ো', 'ৌ', 'ং', 'ঃ', 'ড়', 'ঢ়', 'য়'],
            'Assamese': ['ৰ', 'ৱ', '৲', '৳', '৴', '৵', '৶', '৷', '৸', '৹'],
            'Tamil': ['ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', 'ஃ', 'ஸ', 'ஷ', 'ஜ', 'ஹ', 'க்ஷ'],
            'Telugu': ['ా', 'ి', 'ీ', 'ు', 'ూ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ః', 'క్ష', 'జ్ఞ', 'త్ర', 'శ్ర'],
            'Gujarati': ['ા', 'િ', 'ી', 'ુ', 'ૂ', 'ે', 'ૈ', 'ો', 'ૌ', 'ં', 'ઃ', 'ળ', 'ક્ષ', 'જ્ઞ'],
            'Punjabi': ['ਾ', 'ਿ', 'ੀ', 'ੁ', 'ੂ', 'ੇ', 'ੈ', 'ੋ', 'ੌ', 'ਂ', 'ਃ', 'ਖ਼', 'ਗ਼', 'ਜ਼', 'ੜ', 'ਫ਼'],
            'Malayalam': ['ാ', 'ി', 'ീ', 'ു', 'ൂ', 'െ', 'േ', 'ൈ', 'ൊ', 'ോ', 'ൌ', 'ഃ', 'ക്ഷ', 'ജ്ഞ', 'ശ്ര'],
            'Kannada': ['ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', 'ಂ', 'ಃ', 'ಳ', 'ಕ್ಷ', 'ಜ್ಞ', 'ಶ್ರ'],
            'Odia': ['ା', 'ି', 'ୀ', 'ୁ', 'ୂ', 'େ', 'ୈ', 'ୋ', 'ୌ', 'ଂ', 'ଃ', 'ଳ', 'କ୍ଷ', 'ଜ୍ଞ'],
            'Urdu': ['آ', 'أ', 'إ', 'ئ', 'ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ہ', 'ھ', 'ی', 'ے'],
            'Sanskrit': ['ऋ', 'ॠ', 'ऌ', 'ॡ', 'ऽ', 'ॐ', 'ं', 'ः', '्'],
            'English': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        }
        
        # Script ranges for Indian languages
        self.script_ranges = {
            'Devanagari': (0x0900, 0x097F),  # Hindi, Marathi, Sanskrit, Nepali
            'Bengali': (0x0980, 0x09FF),     # Bengali, Assamese
            'Gurmukhi': (0x0A00, 0x0A7F),    # Punjabi
            'Gujarati': (0x0A80, 0x0AFF),    # Gujarati
            'Odia': (0x0B00, 0x0B7F),        # Odia
            'Tamil': (0x0B80, 0x0BFF),       # Tamil
            'Telugu': (0x0C00, 0x0C7F),      # Telugu
            'Kannada': (0x0C80, 0x0CFF),     # Kannada
            'Malayalam': (0x0D00, 0x0D7F),   # Malayalam
            'Sinhala': (0x0D80, 0x0DFF),     # Sinhala
            'Latin': (0x0041, 0x007A),       # English
            'Arabic': (0x0600, 0x06FF),      # Urdu
        }
        
        # Language to script mapping
        self.language_to_script = {
            'Hindi': 'Devanagari',
            'Marathi': 'Devanagari',
            'Sanskrit': 'Devanagari',
            'Nepali': 'Devanagari',
            'Bengali': 'Bengali',
            'Assamese': 'Bengali',
            'Punjabi': 'Gurmukhi',
            'Gujarati': 'Gujarati',
            'Odia': 'Odia',
            'Tamil': 'Tamil',
            'Telugu': 'Telugu',
            'Kannada': 'Kannada',
            'Malayalam': 'Malayalam',
            'Urdu': 'Arabic',
            'English': 'Latin'
        }
    
    def detect_indian_script(self, text):
        """Detect which Indian script is used in the text"""
        script_counts = {script: 0 for script in self.script_ranges}
        
        for char in text:
            char_code = ord(char)
            for script, (start, end) in self.script_ranges.items():
                if start <= char_code <= end:
                    script_counts[script] += 1
                    break
        
        total_chars = len(text)
        if total_chars > 0:
            script_percentages = {script: count/total_chars for script, count in script_counts.items()}
            dominant_script = max(script_counts.items(), key=lambda x: x[1])[0]
        else:
            script_percentages = {script: 0 for script in self.script_ranges}
            dominant_script = 'Unknown'
        
        return script_percentages, dominant_script
    
    def extract_language_specific_features(self, texts):
        """
        Extract features specific to each Indian language
        Returns a feature matrix with language-specific characteristics
        """
        print("  Extracting language-specific features...")
        
        all_features = []
        
        for text in texts:
            features = []
            
            # 1. Script detection features
            script_percentages, dominant_script = self.detect_indian_script(text)
            features.extend([script_percentages.get('Devanagari', 0),
                           script_percentages.get('Bengali', 0),
                           script_percentages.get('Gurmukhi', 0),
                           script_percentages.get('Gujarati', 0),
                           script_percentages.get('Odia', 0),
                           script_percentages.get('Tamil', 0),
                           script_percentages.get('Telugu', 0),
                           script_percentages.get('Kannada', 0),
                           script_percentages.get('Malayalam', 0),
                           script_percentages.get('Arabic', 0),
                           script_percentages.get('Latin', 0)])
            
            # 2. Language-specific character markers
            for lang, markers in self.language_markers.items():
                marker_count = sum(1 for char in text if char in markers)
                marker_ratio = marker_count / max(len(text), 1)
                features.append(marker_ratio)
            
            # 3. Character statistics
            char_counts = Counter(text)
            total_chars = len(text)
            
            # Character diversity
            features.append(len(char_counts) / max(total_chars, 1))
            
            # Top 5 character frequencies
            if total_chars > 0:
                top_chars = char_counts.most_common(5)
                for char, count in top_chars:
                    features.append(count / total_chars)
                features.extend([0] * (5 - len(top_chars)))
            else:
                features.extend([0] * 5)
            
            # 4. Vowel-consonant ratio for Indian languages
            devanagari_vowels = 'अआइईउऊएऐओऔाीुूेैोौंःऋॠऌॡ'
            tamil_vowels = 'அஆஇஈஉஊஎஏஐஒஓஔாிீுூெேைொோௌ'
            bengali_vowels = 'অআইঈউঊএঐওঔািীুূেৈোৌংঃ'
            
            dev_vowels = sum(1 for c in text if c in devanagari_vowels)
            tam_vowels = sum(1 for c in text if c in tamil_vowels)
            ben_vowels = sum(1 for c in text if c in bengali_vowels)
            
            features.extend([dev_vowels/max(total_chars, 1),
                           tam_vowels/max(total_chars, 1),
                           ben_vowels/max(total_chars, 1)])
            
            # 5. Special character patterns
            matra_count = sum(1 for c in text if c in ['ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ'])  # Hindi matras
            features.append(matra_count / max(total_chars, 1))
            
            # 6. Word-based features
            words = text.split()
            if words:
                avg_word_len = np.mean([len(w) for w in words])
                std_word_len = np.std([len(w) for w in words])
                features.extend([avg_word_len, std_word_len])
            else:
                features.extend([0, 0])
            
            # 7. Unicode block features
            unicode_blocks = {}
            for char in text:
                code = ord(char)
                if 0x0900 <= code <= 0x097F:
                    unicode_blocks['Devanagari'] = unicode_blocks.get('Devanagari', 0) + 1
                elif 0x0980 <= code <= 0x09FF:
                    unicode_blocks['Bengali'] = unicode_blocks.get('Bengali', 0) + 1
                elif 0x0B80 <= code <= 0x0BFF:
                    unicode_blocks['Tamil'] = unicode_blocks.get('Tamil', 0) + 1
                elif 0x0C00 <= code <= 0x0C7F:
                    unicode_blocks['Telugu'] = unicode_blocks.get('Telugu', 0) + 1
            
            for block in ['Devanagari', 'Bengali', 'Tamil', 'Telugu', 'Gujarati', 'Gurmukhi', 'Kannada', 'Malayalam', 'Odia']:
                features.append(unicode_blocks.get(block, 0) / max(total_chars, 1))
            
            all_features.append(features)
        
        return np.array(all_features)
    
    def extract_tfidf_features(self, texts):
        """Extract word-level TF-IDF features"""
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
        return vectorizer.fit_transform(texts)
    
    def extract_char_ngram_features(self, texts, ngram_range=(2, 5)):
        """Extract character n-gram features"""
        vectorizer = TfidfVectorizer(
            max_features=4000,
            ngram_range=ngram_range,
            analyzer='char',
            min_df=2,
            max_df=0.95
        )
        return vectorizer.fit_transform(texts)
    
    def extract_phonetic_features(self, texts):
        """Extract phonetic features specific to Indian languages"""
        features = []
        
        # Vowel sets for different language families
        vowel_sets = {
            'devanagari': 'अआइईउऊएऐओऔाीुूेैोौंःऋॠऌॡ',
            'tamil': 'அஆஇஈஉஊஎஏஐஒஓஔாிீுூெேைொோௌ',
            'bengali': 'অআইঈউঊএঐওঔািীুূেৈোৌংঃ',
            'telugu': 'అఆఇఈఉఊఋఌఎఏఐఒఓఔాిీుూెేైొోౌంః',
            'gujarati': 'અઆઇઈઉઊએઐઓઔાિીુૂેૈોૌંઃ',
            'punjabi': 'ਅਆਇਈਉਊਏਐਓਔਾਿੀੁੂੇੈੋੌਂਃ',
            'malayalam': 'അആഇഈഉഊഋഌഎഏഐഒഓഔാിീുൂെേൈൊോൌംഃ',
            'kannada': 'ಅಆಇಈಉಊಋಌಎಏಐಒಓಔಾಿೀುೂೆೇೈೊೋೌಂಃ',
            'odia': 'ଅଆଇଈଉଊଏଐଓଔାିୀୁୂେୈୋୌଂଃ'
        }
        
        for text in texts:
            text_features = []
            total_chars = len(text)
            
            if total_chars == 0:
                features.append([0] * (len(vowel_sets) + 3))
                continue
            
            # Vowel ratios for each language family
            for lang, vowels in vowel_sets.items():
                vowel_count = sum(1 for c in text if c in vowels)
                text_features.append(vowel_count / total_chars)
            
            # Consonant-vowel ratio
            indian_chars = sum(1 for c in text if any(c in vowels for vowels in vowel_sets.values()))
            vowel_total = sum(text_features[:len(vowel_sets)])
            text_features.append(vowel_total)
            
            # Character entropy (measure of randomness)
            char_counts = Counter(text)
            entropy = -sum((count/total_chars) * np.log2(count/total_chars) 
                          for count in char_counts.values() if count > 0)
            text_features.append(entropy)
            
            # Unique character ratio
            text_features.append(len(char_counts) / total_chars)
            
            features.append(text_features)
        
        return np.array(features)
    
    def extract_structural_features(self, texts):
        """Extract structural features of text"""
        features = []
        
        for text in texts:
            text_features = []
            
            # Length features
            text_len = len(text)
            word_count = len(text.split())
            char_count = len(text.replace(' ', ''))
            
            text_features.extend([text_len, word_count, char_count])
            
            # Ratio features
            if text_len > 0:
                text_features.append(word_count / text_len)  # Word density
                text_features.append(char_count / text_len)  # Character density
            else:
                text_features.extend([0, 0])
            
            # Digit and punctuation features
            digit_count = sum(1 for c in text if c.isdigit())
            punct_count = sum(1 for c in text if c in '.,!?;:"\'')
            
            text_features.extend([digit_count, punct_count])
            if text_len > 0:
                text_features.extend([digit_count/text_len, punct_count/text_len])
            else:
                text_features.extend([0, 0])
            
            # Case features (for English/Urdu mix)
            upper_count = sum(1 for c in text if c.isupper())
            lower_count = sum(1 for c in text if c.islower())
            text_features.extend([upper_count, lower_count])
            
            features.append(text_features)
        
        return np.array(features)
    
    def identify_language_from_text(self, text):
        """
        Identify the most likely language from text using multiple features
        Returns: language name, confidence score
        """
        if not text:
            return "Unknown", 0.0
        
        # Get script information
        script_percentages, dominant_script = self.detect_indian_script(text)
        
        # Check language-specific markers
        marker_scores = {}
        for lang, markers in self.language_markers.items():
            marker_count = sum(1 for char in text if char in markers)
            marker_scores[lang] = marker_count / max(len(text), 1)
        
        # Get script-based language candidates
        script_based_langs = []
        for lang, script in self.language_to_script.items():
            if script == dominant_script:
                script_based_langs.append(lang)
        
        # Score each candidate language
        final_scores = {}
        for lang in script_based_langs:
            # Base score from script
            base_score = script_percentages.get(dominant_script, 0)
            
            # Boost from language markers
            marker_score = marker_scores.get(lang, 0)
            
            # Combined score
            final_scores[lang] = 0.7 * base_score + 0.3 * marker_score
        
        # If no script-based languages, use marker scores
        if not final_scores:
            final_scores = marker_scores
        
        # Get best language
        if final_scores:
            best_lang = max(final_scores.items(), key=lambda x: x[1])
            if best_lang[1] > 0.1:  # Minimum confidence threshold
                return best_lang[0], best_lang[1]
        
        return "Unknown", 0.0
    
    def reduce_dimensions(self, features, method='pca', n_components=100):
        """Reduce feature dimensions"""
        from sklearn.decomposition import PCA, TruncatedSVD
        
        if method == 'pca':
            reducer = PCA(n_components=min(n_components, features.shape[1]), 
                         random_state=42)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=min(n_components, features.shape[1]), 
                                  random_state=42)
        else:
            return features
        
        return reducer.fit_transform(features)
    
    def combine_features(self, feature_matrices):
        """Combine all feature matrices"""
        combined = []
        for matrix in feature_matrices:
            if isinstance(matrix, np.ndarray):
                combined.append(csr_matrix(matrix))
            else:
                combined.append(matrix)
        
        return hstack(combined).toarray()