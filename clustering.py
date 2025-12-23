# src/clustering.py - ENHANCED VERSION
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time

class LanguageClusterer:
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def ensemble_clustering(self, features, n_clusters=14):
        """
        Ensemble clustering for better language separation
        """
        print(f"  Running ensemble clustering with {n_clusters} clusters...")
        
        # Multiple clustering algorithms
        algorithms = {
            'kmeans': KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10),
            'spectral': SpectralClustering(n_clusters=n_clusters, random_state=self.random_state),
            'agglomerative': AgglomerativeClustering(n_clusters=n_clusters)
        }
        
        all_labels = {}
        
        for name, algo in algorithms.items():
            print(f"    Running {name}...", end="", flush=True)
            start_time = time.time()
            
            try:
                if hasattr(algo, 'fit_predict'):
                    labels = algo.fit_predict(features)
                else:
                    labels = algo.fit(features).labels_
                
                all_labels[name] = labels
                elapsed = time.time() - start_time
                print(f" ✓ ({elapsed:.1f}s)")
            except Exception as e:
                print(f" ✗ Failed: {e}")
                all_labels[name] = None
        
        # Consensus clustering (majority vote)
        print("  Performing consensus clustering...")
        consensus_labels = self.consensus_clustering(all_labels, n_clusters)
        
        return consensus_labels
    
    def consensus_clustering(self, all_labels, n_clusters):
        """Combine multiple clustering results"""
        valid_labels = [labels for labels in all_labels.values() if labels is not None]
        
        if not valid_labels:
            return all_labels.get('kmeans', np.zeros(len(all_labels[0])))
        
        # Initialize consensus matrix
        n_samples = len(valid_labels[0])
        consensus_matrix = np.zeros((n_samples, n_samples))
        
        # Build consensus matrix
        for labels in valid_labels:
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        consensus_matrix[i, j] += 1
        
        # Normalize
        consensus_matrix /= len(valid_labels)
        
        # Final clustering on consensus matrix
        final_clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=self.random_state
        )
        
        return final_clustering.fit_predict(consensus_matrix)
    
    def refine_clusters_with_language_rules(self, texts, labels):
        """
        Refine clusters using language-specific rules
        """
        print("  Refining clusters with language rules...")
        
        refined_labels = labels.copy()
        
        for i, text in enumerate(texts):
            script_percentages, dominant_script = self.detect_script(text)
            
            # Language-specific rules
            if dominant_script == 'Devanagari':
                # Check for language-specific characters
                if any(c in text for c in ['ळ', 'श', 'ष']):  # Marathi markers
                    # Ensure it's in a different cluster
                    if labels[i] < 3:  # Assuming clusters 0-2 are Hindi
                        # Move to Marathi cluster
                        refined_labels[i] = max(labels) + 1 if max(labels) < 13 else 3
            
            elif dominant_script == 'Bengali':
                if any(c in text for c in ['ড়', 'ঢ়']):  # Assamese markers
                    refined_labels[i] = 12  # Assamese cluster
                else:
                    refined_labels[i] = 11  # Bengali cluster
            
            # Add more language-specific rules as needed
        
        return refined_labels
    
    def detect_script(self, text):
        """Detect script of text"""
        from feature_extraction import FeatureExtractor
        extractor = FeatureExtractor()
        script_percentages, dominant_script = extractor.detect_indian_script(text)
        return script_percentages, dominant_script
    
    def find_optimal_clusters_enhanced(self, features, max_k=20):
        """Enhanced optimal cluster finding"""
        print("  Finding optimal clusters...")
        
        wcss = []
        silhouette_scores = []
        bic_scores = []
        
        k_range = range(5, max_k + 1, 2)  # Check odd numbers
        
        for k in k_range:
            print(f"    Testing k={k}", end="", flush=True)
            
            # K-Means
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=3)
            kmeans.fit(features)
            wcss.append(kmeans.inertia_)
            
            # Silhouette score
            if k > 1:
                labels = kmeans.labels_
                sil_score = silhouette_score(features, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
            
            print(f" ✓")
        
        # Find elbow point
        diff = np.diff(wcss)
        diff_ratio = diff[1:] / diff[:-1]
        elbow_point = k_range[np.argmin(diff_ratio) + 1]
        
        # Find best silhouette score
        best_silhouette = k_range[np.argmax(silhouette_scores)]
        
        # Combine results (prefer silhouette if good separation)
        if max(silhouette_scores) > 0.3:
            optimal_k = best_silhouette
        else:
            optimal_k = elbow_point
        
        # Constrain to typical Indian languages (14-22)
        optimal_k = max(12, min(optimal_k, 22))
        
        print(f"  ✅ Selected optimal k={optimal_k}")
        
        # Plot
        self.plot_cluster_metrics(k_range, wcss, silhouette_scores, optimal_k)
        
        return optimal_k
    
    def plot_cluster_metrics(self, k_values, wcss, silhouette_scores, optimal_k):
        """Plot clustering metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Elbow plot
        ax1.plot(k_values, wcss, 'bo-')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('WCSS')
        ax1.set_title('Elbow Method')
        ax1.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Silhouette plot
        ax2.plot(k_values, silhouette_scores, 'go-')
        ax2.set_xlabel('Number of clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score')
        ax2.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/cluster_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()