# src/evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import umap
import pickle
from collections import Counter
import textwrap

class UnsupervisedEvaluator:
    def __init__(self):
        """
        Class for evaluating unsupervised clustering results
        """
        pass
    
    def evaluate_clustering(self, features, labels):
        """
        Evaluate clustering quality using internal metrics
        """
        # Remove noise labels (-1) for evaluation
        valid_mask = labels != -1
        if valid_mask.sum() < 2:
            return {}
        
        features_valid = features[valid_mask]
        labels_valid = labels[valid_mask]
        
        # Ensure we have at least 2 clusters
        unique_labels = np.unique(labels_valid)
        if len(unique_labels) < 2:
            return {}
        
        metrics = {
            'silhouette': silhouette_score(features_valid, labels_valid),
            'davies_bouldin': davies_bouldin_score(features_valid, labels_valid),
            'calinski_harabasz': calinski_harabasz_score(features_valid, labels_valid)
        }
        
        print(f"  Silhouette Score: {metrics['silhouette']:.3f}")
        print(f"  Davies-Bouldin Score: {metrics['davies_bouldin']:.3f} (lower is better)")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz']:.1f} (higher is better)")
        
        return metrics
    
    def visualize_clusters(self, features, labels, method_name='K-Means'):
        """Optimized visualization using PCA instead of t-SNE"""
        print("  Creating optimized cluster visualization...")
        
        # Use PCA instead of t-SNE (10x faster)
        from sklearn.decomposition import PCA
        
        # Take a sample for visualization (10k samples max)
        sample_size = min(10000, len(features))
        indices = np.random.choice(len(features), sample_size, replace=False)
        sample_features = features[indices]
        sample_labels = labels[indices]
        
        print(f"  Using {sample_size} samples for visualization...")
        
        # Reduce dimensions with PCA (fast)
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(sample_features)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot clusters
        unique_labels = np.unique(sample_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = sample_labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=[color], alpha=0.7, s=30, label=f'Cluster {label}')
        
        plt.title(f'{method_name} Visualization (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save figure
        output_file = f'visualizations/{method_name.lower().replace(" ", "_")}_clusters_pca.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        print(f"  ✓ Visualization saved to {output_file}")

    def analyze_cluster_coherence(self, texts, labels, top_n_words=20):
        """
        Analyze how coherent each cluster is (similarity within cluster)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        coherence_scores = {}
        
        for label in np.unique(labels):
            if label == -1:
                continue
                
            cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == label]
            
            if len(cluster_texts) < 2:
                coherence_scores[label] = 0
                continue
            
            # Calculate TF-IDF similarity within cluster
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            
            # Average cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(tfidf_matrix)
            np.fill_diagonal(similarities, 0)
            
            if similarities.size > 0:
                avg_similarity = similarities.sum() / (similarities.size - len(similarities))
                coherence_scores[label] = avg_similarity
            else:
                coherence_scores[label] = 0
        
        print("\nCluster Coherence Analysis:")
        for label, coherence in sorted(coherence_scores.items(), key=lambda x: x[1], reverse=True):
            cluster_size = sum(labels == label)
            print(f"  Cluster {label}: coherence={coherence:.3f}, size={cluster_size}")
        
        return coherence_scores
    
    def generate_cluster_report(self, texts, labels, output_file='results/cluster_report.txt'):
        """
        Generate detailed report for each cluster
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("LANGUAGE CLUSTER ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            unique_labels = np.unique(labels[labels != -1])
            
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                cluster_texts = [texts[i] for i in cluster_indices]
                cluster_size = len(cluster_texts)
                
                f.write(f"\n{'='*50}\n")
                f.write(f"CLUSTER {label} (Size: {cluster_size})\n")
                f.write(f"{'='*50}\n\n")
                
                # Sample texts
                f.write("SAMPLE TEXTS:\n")
                for i, text in enumerate(cluster_texts[:5]):
                    wrapped_text = textwrap.fill(text, width=80)
                    f.write(f"{i+1}. {wrapped_text}\n")
                
                # Character analysis
                all_chars = ''.join(cluster_texts)
                char_counts = Counter(all_chars)
                
                f.write(f"\nCHARACTER ANALYSIS:\n")
                f.write(f"Total characters: {len(all_chars)}\n")
                f.write(f"Unique characters: {len(char_counts)}\n")
                f.write(f"Top 10 characters: {char_counts.most_common(10)}\n")
                
                # Script detection
                scripts = {
                    'Devanagari': sum(1 for c in all_chars if '\u0900' <= c <= '\u097F'),
                    'Bengali': sum(1 for c in all_chars if '\u0980' <= c <= '\u09FF'),
                    'Tamil': sum(1 for c in all_chars if '\u0B80' <= c <= '\u0BFF'),
                    'Telugu': sum(1 for c in all_chars if '\u0C00' <= c <= '\u0C7F'),
                    'Latin': sum(1 for c in all_chars if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
                }
                
                f.write(f"\nSCRIPT DISTRIBUTION:\n")
                for script, count in scripts.items():
                    if count > 0:
                        percentage = (count / len(all_chars)) * 100 if all_chars else 0
                        f.write(f"  {script}: {count} ({percentage:.1f}%)\n")
                
                # Word analysis (if spaces exist)
                if any(' ' in text for text in cluster_texts):
                    words = ' '.join(cluster_texts).split()
                    word_counts = Counter(words)
                    f.write(f"\nWORD ANALYSIS:\n")
                    f.write(f"Total words: {len(words)}\n")
                    f.write(f"Unique words: {len(word_counts)}\n")
                    f.write(f"Top 10 words: {word_counts.most_common(10)}\n")
                
                f.write("\n")
        
            # Noise analysis
            noise_indices = np.where(labels == -1)[0]
            if len(noise_indices) > 0:
                f.write(f"\n{'='*50}\n")
                f.write(f"NOISE POINTS (Unclustered) - {len(noise_indices)} texts\n")
                f.write(f"{'='*50}\n\n")
                
                for i in noise_indices[:10]:
                    f.write(f"- {texts[i][:100]}...\n")
        
        print(f"Cluster report saved to: {output_file}")
    
    def compare_clustering_methods(self, features, all_labels):
        """
        Compare different clustering methods
        """
        comparison_data = []
        
        for method_name, labels in all_labels.items():
            if labels is None or len(set(labels[labels != -1])) < 2:
                continue
            
            metrics = self.evaluate_clustering(features, labels)
            if metrics:
                comparison_data.append({
                    'Method': method_name,
                    'Silhouette': metrics['silhouette'],
                    'Davies-Bouldin': metrics['davies_bouldin'],
                    'Calinski-Harabasz': metrics['calinski_harabasz']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Plot comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            metrics_to_plot = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
            titles = ['Silhouette Score (higher better)', 
                     'Davies-Bouldin Score (lower better)', 
                     'Calinski-Harabasz Score (higher better)']
            
            for ax, metric, title in zip(axes, metrics_to_plot, titles):
                ax.bar(comparison_df['Method'], comparison_df[metric])
                ax.set_title(title)
                ax.set_xticklabels(comparison_df['Method'], rotation=45)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('visualizations/clustering_comparison.png', dpi=300)
            plt.show()
            
            return comparison_df
        
        return None