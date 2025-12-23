# main.py - COMPLETE WORKING VERSION
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import re
import unicodedata

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Define FeatureExtractor class
class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        
        # Language-specific character markers
        self.language_markers = {
            'Hindi': ['ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', 'ऋ'],
            'Marathi': ['ळ', 'श्र', 'क्क', 'त्त', 'द्ध'],
            'Bengali': ['া', 'ি', 'ী', 'ু', 'ূ', 'ে', 'ৈ', 'ো', 'ৌ', 'ং', 'ঃ', 'ড়', 'ঢ়'],
            'Assamese': ['ৰ', 'ৱ', '৲', '৳'],
            'Tamil': ['ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', 'ஃ'],
            'Telugu': ['ా', 'ి', 'ీ', 'ు', 'ూ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ః'],
            'Gujarati': ['ા', 'િ', 'ી', 'ુ', 'ૂ', 'ે', 'ૈ', 'ો', 'ૌ', 'ં', 'ઃ', 'ળ'],
            'Punjabi': ['ਾ', 'ਿ', 'ੀ', 'ੁ', 'ੂ', 'ੇ', 'ੈ', 'ੋ', 'ੌ', 'ਂ', 'ਃ', 'ਖ਼', 'ਗ਼'],
            'Malayalam': ['ാ', 'ി', 'ീ', 'ു', 'ൂ', 'െ', 'േ', 'ൈ', 'ൊ', 'ോ', 'ൌ', 'ഃ'],
            'Kannada': ['ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', 'ಂ', 'ಃ', 'ಳ'],
            'Odia': ['ା', 'ି', 'ୀ', 'ୁ', 'ୂ', 'େ', 'ୈ', 'ୋ', 'ୌ', 'ଂ', 'ଃ', 'ଳ'],
            'Urdu': ['آ', 'أ', 'إ', 'ئ', 'ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ہ', 'ھ', 'ی', 'ے'],
            'Sanskrit': ['ऋ', 'ॠ', 'ऌ', 'ॡ', 'ऽ', 'ॐ'],
            'English': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        }
        
        # Script ranges for Indian languages
        self.script_ranges = {
            'Devanagari': (0x0900, 0x097F),
            'Bengali': (0x0980, 0x09FF),
            'Gurmukhi': (0x0A00, 0x0A7F),
            'Gujarati': (0x0A80, 0x0AFF),
            'Odia': (0x0B00, 0x0B7F),
            'Tamil': (0x0B80, 0x0BFF),
            'Telugu': (0x0C00, 0x0C7F),
            'Kannada': (0x0C80, 0x0CFF),
            'Malayalam': (0x0D00, 0x0D7F),
            'Sinhala': (0x0D80, 0x0DFF),
            'Latin': (0x0041, 0x007A),
            'Arabic': (0x0600, 0x06FF),
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
        """Extract features specific to each Indian language"""
        all_features = []
        
        for text in texts:
            features = []
            
            # Script detection features
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
            
            # Language-specific character markers
            for lang, markers in self.language_markers.items():
                marker_count = sum(1 for char in text if char in markers)
                marker_ratio = marker_count / max(len(text), 1)
                features.append(marker_ratio)
            
            # Character statistics
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
            
            # Vowel-consonant ratio indicators
            devanagari_vowels = 'अआइईउऊएऐओऔाीुूेैोौंःऋॠऌॡ'
            tamil_vowels = 'அஆஇஈஉஊஎஏஐஒஓஔாிீுூெேைொோௌ'
            bengali_vowels = 'অআইঈউঊএঐওঔািীুূেৈোৌংঃ'
            
            dev_vowels = sum(1 for c in text if c in devanagari_vowels)
            tam_vowels = sum(1 for c in text if c in tamil_vowels)
            ben_vowels = sum(1 for c in text if c in bengali_vowels)
            
            features.extend([dev_vowels/max(total_chars, 1),
                           tam_vowels/max(total_chars, 1),
                           ben_vowels/max(total_chars, 1)])
            
            # Special character patterns
            matra_count = sum(1 for c in text if c in ['ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ'])
            features.append(matra_count / max(total_chars, 1))
            
            # Word-based features
            words = text.split()
            if words:
                avg_word_len = np.mean([len(w) for w in words])
                std_word_len = np.std([len(w) for w in words])
                features.extend([avg_word_len, std_word_len])
            else:
                features.extend([0, 0])
            
            all_features.append(features)
        
        return np.array(all_features)

    def extract_tfidf_features(self, texts):
        """Extract word-level TF-IDF features"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            analyzer='word'
        )
        return vectorizer.fit_transform(texts)

    def extract_char_ngram_features(self, texts):
        """Extract character n-gram features"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(2, 4),
            analyzer='char',
            min_df=2,
            max_df=0.95
        )
        return vectorizer.fit_transform(texts)

    def identify_language_from_text(self, text):
        """Identify the most likely language from text"""
        if not text or len(text) < 5:
            return "Unknown", 0.0
        
        # Get script information
        script_percentages, dominant_script = self.detect_indian_script(text)
        
        # Language to script mapping
        script_to_languages = {
            'Devanagari': ['Hindi', 'Marathi', 'Sanskrit', 'Nepali'],
            'Bengali': ['Bengali', 'Assamese'],
            'Gurmukhi': ['Punjabi'],
            'Gujarati': ['Gujarati'],
            'Odia': ['Odia'],
            'Tamil': ['Tamil'],
            'Telugu': ['Telugu'],
            'Kannada': ['Kannada'],
            'Malayalam': ['Malayalam'],
            'Arabic': ['Urdu'],
            'Latin': ['English']
        }
        
        # Get candidate languages for this script
        candidate_langs = script_to_languages.get(dominant_script, [])
        
        if not candidate_langs:
            return "Unknown", 0.0
        
        # Score each candidate language
        scores = {}
        for lang in candidate_langs:
            if lang in self.language_markers:
                markers = self.language_markers[lang]
                marker_count = sum(1 for char in text if char in markers)
                score = marker_count / max(len(text), 1)
                scores[lang] = score
        
        if scores:
            best_lang = max(scores.items(), key=lambda x: x[1])
            if best_lang[1] > 0.1:  # Minimum confidence
                return best_lang[0], best_lang[1]
        
        # Fallback to script-based identification
        if dominant_script in script_to_languages:
            return script_to_languages[dominant_script][0], 0.5
        
        return "Unknown", 0.0

# Define TextPreprocessor class
class TextPreprocessor:
    def __init__(self):
        pass
    
    def clean_text(self, text):
        """Clean and normalize text for Indian languages"""
        if not isinstance(text, str):
            return ""
        
        text = str(text).strip()
        if not text:
            return ""
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Keep all Indian language characters
        text = re.sub(r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def get_text_statistics(self, texts):
        """Get basic statistics about the text data"""
        if not texts:
            return {'avg_text_length': 0, 'min_text_length': 0, 'max_text_length': 0, 'unique_characters': 0}
        
        lengths = [len(text) for text in texts if text]
        
        return {
            'avg_text_length': np.mean(lengths) if lengths else 0,
            'min_text_length': min(lengths) if lengths else 0,
            'max_text_length': max(lengths) if lengths else 0,
            'unique_characters': len(set(''.join(texts))) if texts else 0,
        }

class IndianLanguageIdentifier:
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.clusters = {}
    
    def load_data(self):
        """Load datasets"""
        print("="*60)
        print("STEP 1: LOADING DATASETS")
        print("="*60)
        
        if not os.path.exists(self.data_path):
            print(f"❌ Data directory '{self.data_path}' not found!")
            os.makedirs(self.data_path, exist_ok=True)
            return None
        
        all_texts = []
        
        # Look for CSV files
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        
        if not csv_files:
            print("❌ No CSV files found in data/ directory")
            return None
        
        for csv_file in csv_files[:3]:  # Process up to 3 CSV files
            file_path = os.path.join(self.data_path, csv_file)
            try:
                print(f"Loading {csv_file}...")
                df = pd.read_csv(file_path, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(file_path, encoding='latin1')
                except Exception as e:
                    print(f"  Failed to load {csv_file}: {e}")
                    continue
            
            # Try to find text column
            text_columns = [col for col in df.columns if 'text' in col.lower() or 
                           'headline' in col.lower() or 'sentence' in col.lower()]
            
            if text_columns:
                text_col = text_columns[0]
                texts = df[text_col].dropna().astype(str).tolist()
                all_texts.extend(texts)
                print(f"  Found {len(texts)} texts in column '{text_col}'")
            else:
                # Try first column
                try:
                    texts = df.iloc[:, 0].dropna().astype(str).tolist()
                    all_texts.extend(texts)
                    print(f"  Found {len(texts)} texts in first column")
                except:
                    continue
        
        if not all_texts:
            print("❌ No texts found in CSV files")
            return None
        
        print(f"\n✅ Total texts loaded: {len(all_texts):,}")
        print("Sample texts:")
        for i in range(min(3, len(all_texts))):
            print(f"  {i+1}. {all_texts[i][:100]}...")
        
        return all_texts
    
    def preprocess_data(self, texts):
        """Preprocess text data"""
        print("\n" + "="*60)
        print("STEP 2: PREPROCESSING TEXT DATA")
        print("="*60)
        
        print(f"Cleaning {len(texts):,} texts...")
        cleaned_texts = []
        
        for i, text in enumerate(texts):
            cleaned = self.preprocessor.clean_text(text)
            if cleaned and len(cleaned) >= 10:
                cleaned_texts.append(cleaned)
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1:,} texts...")
        
        print(f"  ✅ Finished cleaning {len(cleaned_texts):,} texts")
        
        stats = self.preprocessor.get_text_statistics(cleaned_texts)
        print(f"\n📊 Text Statistics:")
        print(f"  Average length: {stats['avg_text_length']:.1f} characters")
        print(f"  Min length: {stats['min_text_length']} characters")
        print(f"  Max length: {stats['max_text_length']} characters")
        print(f"  Unique characters: {stats['unique_characters']}")
        
        return cleaned_texts
    
    def extract_features(self, texts):
        """Extract features for clustering"""
        print("\n" + "="*60)
        print("STEP 3: FEATURE EXTRACTION")
        print("="*60)
        
        print(f"Processing {len(texts):,} texts...")
        start_time = time.time()
        
        # Extract different feature types
        print("1. Extracting language-specific features...")
        lang_features = self.feature_extractor.extract_language_specific_features(texts)
        print(f"   ✓ Shape: {lang_features.shape}")
        
        print("2. Extracting TF-IDF features...")
        tfidf_features = self.feature_extractor.extract_tfidf_features(texts)
        print(f"   ✓ Shape: {tfidf_features.shape}")
        
        print("3. Extracting character n-gram features...")
        char_features = self.feature_extractor.extract_char_ngram_features(texts)
        print(f"   ✓ Shape: {char_features.shape}")
        
        # Combine features
        print("4. Combining all features...")
        from scipy.sparse import hstack, csr_matrix
        
        # Convert to sparse matrices
        tfidf_sparse = tfidf_features
        char_sparse = char_features
        lang_sparse = csr_matrix(lang_features)
        
        # Combine all features
        combined_sparse = hstack([tfidf_sparse, char_sparse, lang_sparse])
        all_features = combined_sparse.toarray()
        
        print(f"   ✓ Combined shape: {all_features.shape}")
        
        # Reduce dimensions
        print("5. Reducing dimensions...")
        from sklearn.decomposition import PCA
        
        n_components = min(100, all_features.shape[1], len(texts) - 1)
        pca = PCA(n_components=n_components, random_state=42)
        reduced_features = pca.fit_transform(all_features)
        
        print(f"   ✓ Reduced to {reduced_features.shape[1]} dimensions")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Feature extraction completed in {elapsed:.1f} seconds")
        
        return {
            'full_features': all_features,
            'reduced_features': reduced_features,
            'texts': texts
        }
    
    def perform_clustering(self, features_dict):
        """Perform clustering"""
        print("\n" + "="*60)
        print("STEP 4: CLUSTERING")
        print("="*60)
        
        if features_dict is None:
            print("❌ No features available for clustering")
            return None
        
        reduced_features = features_dict['reduced_features']
        texts = features_dict['texts']
        
        # For Indian languages, we target 14 clusters (major languages)
        target_clusters = 14
        
        print(f"1. Performing K-Means clustering with k={target_clusters}...")
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(
            n_clusters=target_clusters,
            random_state=42,
            n_init=10,
            max_iter=200,
            verbose=0
        )
        
        labels = kmeans.fit_predict(reduced_features)
        
        # Analyze cluster distribution
        cluster_sizes = Counter(labels)
        print(f"\n📊 Cluster Distribution:")
        for cluster_id in sorted(cluster_sizes.keys()):
            count = cluster_sizes[cluster_id]
            percentage = (count / len(labels)) * 100
            print(f"  Cluster {cluster_id}: {count:,} texts ({percentage:.1f}%)")
        
        # Map clusters to languages
        print("\n2. Mapping clusters to languages...")
        language_mapping = self.map_clusters_to_languages(texts, labels)
        
        # Analyze each cluster
        print("\n3. Analyzing cluster characteristics...")
        cluster_stats = self.analyze_clusters(texts, labels, language_mapping)
        
        # Store results
        self.clusters = {
            'labels': labels,
            'model': kmeans,
            'n_clusters': target_clusters,
            'language_mapping': language_mapping,
            'cluster_stats': cluster_stats,
            'texts': texts,
            'features': reduced_features
        }
        
        # Save results
        os.makedirs('results', exist_ok=True)
        with open('results/clustering_results.pkl', 'wb') as f:
            pickle.dump(self.clusters, f)
        print("\n✅ Clustering results saved to results/clustering_results.pkl")
        
        return self.clusters
    
    def map_clusters_to_languages(self, texts, labels):
        """Map each cluster to a specific Indian language"""
        language_mapping = {}
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            if cluster_size < 50:
                language_mapping[cluster_id] = f"Small Cluster {cluster_id}"
                continue
            
            # Sample texts for analysis
            sample_size = min(100, cluster_size)
            sample_indices = np.random.choice(cluster_indices, sample_size, replace=False)
            sample_texts = [texts[i] for i in sample_indices]
            
            # Identify language for each sample
            language_votes = Counter()
            
            for text in sample_texts:
                language, confidence = self.feature_extractor.identify_language_from_text(text)
                if language != "Unknown":
                    language_votes[language] += 1
            
            # Determine cluster language
            if language_votes:
                top_language = language_votes.most_common(1)[0][0]
                top_count = language_votes.most_common(1)[0][1]
                
                # Calculate confidence
                confidence = top_count / sample_size
                
                if confidence > 0.5:  # Clear majority
                    language_mapping[cluster_id] = top_language
                else:
                    # Get top 2 languages
                    top2 = language_votes.most_common(2)
                    if len(top2) == 2:
                        language_mapping[cluster_id] = f"{top2[0][0]}/{top2[1][0]}"
                    else:
                        language_mapping[cluster_id] = top_language
            else:
                # Try to identify by script
                script_counts = Counter()
                for text in sample_texts[:20]:
                    _, dominant_script = self.feature_extractor.detect_indian_script(text)
                    script_counts[dominant_script] += 1
                
                if script_counts:
                    dominant_script = script_counts.most_common(1)[0][0]
                    
                    # Map script to likely language
                    script_to_lang = {
                        'Devanagari': 'Hindi',
                        'Bengali': 'Bengali',
                        'Tamil': 'Tamil',
                        'Telugu': 'Telugu',
                        'Gujarati': 'Gujarati',
                        'Gurmukhi': 'Punjabi',
                        'Kannada': 'Kannada',
                        'Malayalam': 'Malayalam',
                        'Odia': 'Odia',
                        'Arabic': 'Urdu',
                        'Latin': 'English'
                    }
                    
                    language_mapping[cluster_id] = script_to_lang.get(dominant_script, dominant_script)
                else:
                    language_mapping[cluster_id] = f"Unknown Cluster {cluster_id}"
        
        return language_mapping
    
    def analyze_clusters(self, texts, labels, language_mapping):
        """Analyze characteristics of each cluster"""
        cluster_stats = {}
        
        for cluster_id in np.unique(labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            if cluster_size == 0:
                continue
            
            # Sample texts for analysis
            sample_size = min(50, cluster_size)
            sample_indices = np.random.choice(cluster_indices, sample_size, replace=False)
            sample_texts = [texts[i] for i in sample_indices]
            
            # Basic statistics
            avg_length = np.mean([len(text) for text in sample_texts])
            
            # Script analysis
            script_counts = Counter()
            for text in sample_texts:
                _, dominant_script = self.feature_extractor.detect_indian_script(text)
                script_counts[dominant_script] += 1
            
            dominant_script = script_counts.most_common(1)[0][0] if script_counts else "Unknown"
            script_percentage = (script_counts[dominant_script] / sample_size) * 100 if script_counts else 0
            
            cluster_stats[cluster_id] = {
                'size': cluster_size,
                'avg_length': avg_length,
                'dominant_script': dominant_script,
                'script_percentage': script_percentage,
                'assigned_language': language_mapping.get(cluster_id, "Unknown"),
                'sample_text': sample_texts[0][:100] if sample_texts else ""
            }
        
        return cluster_stats
    
    def evaluate_clusters(self):
        """Evaluate clustering quality"""
        print("\n" + "="*60)
        print("STEP 5: EVALUATION")
        print("="*60)
        
        if not self.clusters:
            print("❌ No clusters to evaluate")
            return
        
        labels = self.clusters['labels']
        features = self.clusters['features']
        language_mapping = self.clusters['language_mapping']
        
        # Calculate metrics
        print("1. Calculating clustering metrics...")
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        # Remove noise points if any
        valid_mask = labels != -1
        if valid_mask.sum() > 1:
            features_valid = features[valid_mask]
            labels_valid = labels[valid_mask]
            
            if len(set(labels_valid)) > 1:
                silhouette = silhouette_score(features_valid, labels_valid)
                db_score = davies_bouldin_score(features_valid, labels_valid)
                
                print(f"   Silhouette Score: {silhouette:.3f}")
                print(f"   Davies-Bouldin Score: {db_score:.3f}")
                
                # Save evaluation results
                eval_results = {
                    'silhouette': silhouette,
                    'davies_bouldin': db_score,
                    'n_clusters': len(set(labels)),
                    'cluster_sizes': np.bincount(labels[labels >= 0])
                }
                
                with open('results/evaluation_results.pkl', 'wb') as f:
                    pickle.dump(eval_results, f)
                print("✅ Evaluation results saved")
        
        # Generate language distribution report
        print("\n2. Generating language distribution report...")
        self.generate_language_report()
        
        # Create visualizations
        print("\n3. Creating visualizations...")
        self.create_visualizations()
    
    def generate_language_report(self):
        """Generate language distribution report"""
        if not self.clusters:
            return
        
        labels = self.clusters['labels']
        language_mapping = self.clusters['language_mapping']
        
        # Count texts per language
        language_counts = {}
        for label in np.unique(labels):
            if label == -1:
                language_name = "Noise/Unclustered"
            else:
                language_name = language_mapping.get(label, f"Cluster {label}")
            
            count = np.sum(labels == label)
            language_counts[language_name] = count
        
        # Sort by count
        sorted_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Save report
        report_path = 'results/language_distribution.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("LANGUAGE DISTRIBUTION REPORT\n")
            f.write("="*60 + "\n\n")
            
            total_texts = len(labels)
            f.write(f"Total texts analyzed: {total_texts:,}\n")
            f.write(f"Number of clusters: {len(set(labels))}\n\n")
            
            f.write("Language Distribution:\n")
            f.write("-"*40 + "\n")
            
            for language, count in sorted_languages:
                percentage = (count / total_texts) * 100
                f.write(f"{language:25s}: {count:8,} texts ({percentage:5.1f}%)\n")
        
        print(f"   ✓ Language report saved to {report_path}")
        
        # Create visualization
        try:
            plt.figure(figsize=(12, 6))
            languages = [lang for lang, _ in sorted_languages[:15]]
            counts = [count for _, count in sorted_languages[:15]]
            
            bars = plt.barh(languages, counts)
            plt.xlabel('Number of Texts')
            plt.title('Language Distribution (Top 15)')
            plt.gca().invert_yaxis()
            
            # Add count labels
            for bar, count in zip(bars, counts):
                plt.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{count:,}', va='center')
            
            plt.tight_layout()
            os.makedirs('visualizations', exist_ok=True)
            plt.savefig('visualizations/language_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Visualization saved to visualizations/language_distribution.png")
        except Exception as e:
            print(f"   ⚠️ Could not create visualization: {e}")
    
    def create_visualizations(self):
        """Create cluster visualizations"""
        try:
            from sklearn.decomposition import PCA
            import matplotlib.cm as cm
            
            if not self.clusters:
                return
            
            features = self.clusters['features']
            labels = self.clusters['labels']
            
            # Reduce to 2D for visualization
            pca = PCA(n_components=2, random_state=42)
            features_2d = pca.fit_transform(features)
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            
            unique_labels = np.unique(labels)
            colors = cm.tab20(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=[color], alpha=0.6, s=20, label=f'Cluster {label}')
            
            plt.title('Language Clusters Visualization (PCA)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            plt.savefig('visualizations/cluster_visualization.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Cluster visualization saved to visualizations/cluster_visualization.png")
        except Exception as e:
            print(f"   ⚠️ Could not create cluster visualization: {e}")
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("="*60)
        print("INDIAN LANGUAGE IDENTIFICATION PIPELINE")
        print("="*60)
        
        total_start = time.time()
        
        try:
            # Step 1: Load data
            all_texts = self.load_data()
            if all_texts is None:
                return
            
            # Step 2: Preprocess
            cleaned_texts = self.preprocess_data(all_texts)
            if not cleaned_texts:
                print("❌ No texts after preprocessing")
                return
            
            # Step 3: Extract features
            features_dict = self.extract_features(cleaned_texts)
            if features_dict is None:
                return
            
            # Step 4: Perform clustering
            clusters = self.perform_clustering(features_dict)
            if clusters is None:
                return
            
            # Step 5: Evaluate
            self.evaluate_clusters()
            
            total_time = time.time() - total_start
            
            print("\n" + "="*60)
            print(f"✅ PIPELINE COMPLETED IN {total_time/60:.1f} MINUTES!")
            print("="*60)
            
            # Final summary
            if self.clusters:
                labels = self.clusters['labels']
                unique_clusters = len(set(labels))
                total_texts = len(labels)
                
                print(f"\n🎉 FINAL RESULTS:")
                print(f"   Total texts processed: {total_texts:,}")
                print(f"   Clusters identified: {unique_clusters}")
                print(f"   Average cluster size: {total_texts/unique_clusters:.0f}")
                
                # Show top languages
                language_mapping = self.clusters['language_mapping']
                language_counts = {}
                for label in np.unique(labels):
                    if label != -1:
                        lang = language_mapping.get(label, "Unknown")
                        language_counts[lang] = language_counts.get(lang, 0) + np.sum(labels == label)
                
                top_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\n🏆 Top 5 Languages Identified:")
                for i, (lang, count) in enumerate(top_languages, 1):
                    percentage = (count / total_texts) * 100
                    print(f"   {i}. {lang}: {count:,} texts ({percentage:.1f}%)")
            
            print(f"\n📁 Results saved in:")
            print(f"   results/clustering_results.pkl")
            print(f"   results/language_distribution.txt")
            print(f"   results/evaluation_results.pkl")
            print(f"   visualizations/")
            
            print(f"\n🚀 Next: Run 'streamlit run app.py' for interactive exploration")
            
        except KeyboardInterrupt:
            print("\n⚠️ Process interrupted by user")
        except Exception as e:
            print(f"\n❌ Error in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Run the pipeline
    print("\n" + "="*60)
    print("STARTING INDIAN LANGUAGE IDENTIFICATION")
    print("="*60)
    
    project = IndianLanguageIdentifier()
    project.run_pipeline()