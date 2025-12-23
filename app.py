# app.py - FIXED VISIBILITY & SIMPLIFIED
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

# Page config
st.set_page_config(
    page_title="Indian Language Identification",
    page_icon="🇮🇳",
    layout="wide"
)

# FIXED CSS - Better text visibility
st.markdown("""
<style>
/* ===== App Background ===== */
.stApp {
    background-color: #FFFFFF;
}

/* ===== Headings ===== */
h1, h2, h3, h4 {
    color: #1E3A8A; /* Blue */
}

/* ===== General Text ===== */
p, span, label, div {
    color: #1E3A8A; /* Blue text */
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background-color: #ECFEFF; /* light blue */
    border-right: 3px solid #1E3A8A;
}

/* ===== Metrics ===== */
.stMetric {
    background-color: #FFFFFF;
    border: 2px solid #16A34A; /* Green */
    border-radius: 10px;
    padding: 12px;
}

/* ===== DataFrames ===== */
.stDataFrame {
    background-color: #FFFFFF;
    border: 2px solid #16A34A;
}

/* ===== Text Inputs ===== */
textarea, input {
    background-color: #FFFFFF !important;
    color: #1E3A8A !important;
    border: 2px solid #16A34A !important;
    border-radius: 8px;
}

/* ===== Primary Buttons ===== */
button[kind="primary"] {
    background-color: #F97316 !important; /* Orange */
    color: #FFFFFF !important;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

/* ===== Secondary Buttons ===== */
button {
    border-radius: 8px;
    border: 2px solid #1E3A8A;
    color: #1E3A8A;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab"] {
    color: #1E3A8A;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #16A34A !important; /* Green */
    color: #FFFFFF !important;
    border-radius: 8px;
}

/* ===== Expanders ===== */
details {
    background-color: #F0FDF4; /* very light green */
    border: 2px solid #16A34A;
    border-radius: 8px;
    padding: 6px;
}

/* ===== Charts ===== */
svg text {
    fill: #1E3A8A !important;
}

/* ===== Footer ===== */
footer {
    color: #1E3A8A;
}
</style>
""", unsafe_allow_html=True)


# Title
st.markdown('<h1 class="main-title">🇮🇳 Indian Language Identification Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analyze clustered Indian language text data</p>', unsafe_allow_html=True)

# Sidebar - SIMPLIFIED
with st.sidebar:
    st.title("⚙️ Settings")
    
    with st.expander("🤖 Model Info", expanded=True):
        st.write("**Algorithm:** K-Means Clustering")
        st.write("**Features:** Script patterns")
        st.write("**Type:** Unsupervised (no labels)")
    
    st.markdown("---")
    st.title("🔍 Filters")
    
    min_size = st.slider("Min Cluster Size", 50, 5000, 500, 50)
    show_noise = st.checkbox("Show Noise", value=False)
    
    st.markdown("---")
    st.title("📊 Display")
    
    show_samples = st.slider("Samples per Cluster", 1, 10, 3)
    show_stats = st.checkbox("Show Statistics", value=True)

# Load data
def load_data():
    try:
        files = ['results/clustering_results.pkl', 
                'results/indian_clustering_results.pkl',
                'results/final_clusters.pkl']
        
        for file in files:
            if os.path.exists(file):
                with open(file, 'rb') as f:
                    clusters = pickle.load(f)
                return clusters, file
        return None, None
    except:
        return None, None

# Load data
clusters, loaded_from = load_data()

# MAIN APP
if clusters is None:
    st.error("❌ **No clustering results found**")
    st.write("Please run the clustering pipeline first:")
    st.code("python main.py", language="bash")
    
    with st.expander("📁 Check your files"):
        if os.path.exists('results'):
            files = os.listdir('results')
            if files:
                st.write("**Files found:**")
                for f in files:
                    st.write(f"- {f}")
            else:
                st.write("results folder is empty")
        else:
            st.write("results folder does not exist")
    
else:
    # Show loaded info
    st.info(f"✅ **Loaded from:** {loaded_from}")
    
    # Extract data
    labels = clusters['labels']
    texts = clusters['texts']
    mapping = clusters.get('language_mapping', {})
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Clusters", "🎯 Analyze"])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.title("📊 Clustering Overview")
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Texts", len(texts))
        with col2:
            actual_clusters = len(np.unique(labels))
            st.metric("Clusters", actual_clusters)
        with col3:
            avg_size = np.mean([np.sum(labels == c) for c in np.unique(labels) if c != -1])
            st.metric("Avg Size", f"{avg_size:.0f}")
        with col4:
            if -1 in labels:
                noise = np.sum(labels == -1)
                st.metric("Noise", noise)
            else:
                st.metric("Noise", 0)
        
        # Cluster sizes chart
        st.markdown("---")
        st.subheader("📈 Cluster Sizes")
        
        # Get cluster sizes
        cluster_sizes = []
        cluster_names = []
        
        for c in np.unique(labels):
            if c == -1 and not show_noise:
                continue
            
            size = np.sum(labels == c)
            if size < min_size:
                continue
            
            if c == -1:
                name = "Noise"
            else:
                name = mapping.get(c, f"Cluster {c}")
            
            cluster_sizes.append(size)
            cluster_names.append(name)
        
        if cluster_sizes:
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(cluster_names, cluster_sizes, color='#3B82F6')
            ax.set_xlabel('Number of Texts')
            ax.set_title('Cluster Sizes')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + max(cluster_sizes)*0.01, 
                       bar.get_y() + bar.get_height()/2,
                       f'{int(width):,}', va='center')
            
            st.pyplot(fig)
        else:
            st.warning("No clusters meet the minimum size requirement")
        
        # Language distribution
        st.markdown("---")
        st.subheader("🇮🇳 Language Distribution")
        
        # Count by language
        lang_counts = {}
        for c in np.unique(labels):
            if c == -1:
                continue
            
            lang = mapping.get(c, "Unknown")
            if "/" in lang:
                lang = lang.split("/")[0]
            
            size = np.sum(labels == c)
            if size >= min_size:
                lang_counts[lang] = lang_counts.get(lang, 0) + size
        
        if lang_counts:
            # Show as dataframe
            df = pd.DataFrame({
                'Language': list(lang_counts.keys()),
                'Count': list(lang_counts.values()),
                'Percentage': [f"{(v/len(texts)*100):.1f}%" for v in lang_counts.values()]
            }).sort_values('Count', ascending=False)
            
            st.dataframe(df, use_container_width=True)
    
    # TAB 2: CLUSTERS
    with tab2:
        st.title("🔍 Explore Clusters")
        
        # Filter clusters
        all_clusters = [c for c in np.unique(labels) if c != -1]
        filtered_clusters = [c for c in all_clusters if np.sum(labels == c) >= min_size]
        
        if not filtered_clusters:
            st.warning("No clusters meet the minimum size requirement")
        else:
            # Cluster selector
            selected = st.selectbox(
                "Select cluster to explore:",
                filtered_clusters,
                format_func=lambda x: f"Cluster {x}: {mapping.get(x, 'Unknown')} ({np.sum(labels == x):,} texts)",
                index=0
            )
            
            if selected:
                indices = np.where(labels == selected)[0]
                cluster_texts = [texts[i] for i in indices]
                cluster_size = len(cluster_texts)
                
                # Cluster header
                st.markdown(f"### Cluster {selected}: {mapping.get(selected, 'Unknown')}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Size", f"{cluster_size:,}")
                with col2:
                    st.metric("Percentage", f"{(cluster_size/len(texts)*100):.1f}%")
                with col3:
                    avg_len = np.mean([len(t) for t in cluster_texts[:100]]) if cluster_texts else 0
                    st.metric("Avg Length", f"{avg_len:.0f}")
                
                # Sample texts - FIXED VISIBILITY
                st.markdown("---")
                st.subheader("📝 Sample Texts")
                
                if cluster_texts:
                    # Show samples in clear boxes
                    samples_to_show = min(show_samples, len(cluster_texts))
                    sample_indices = np.random.choice(len(cluster_texts), samples_to_show, replace=False)
                    
                    for i, idx in enumerate(sample_indices):
                        text = cluster_texts[idx]
                        with st.expander(f"Sample {i+1} ({len(text)} characters)", expanded=True):
                            # FIX: Use markdown with custom class for clear visibility
                            st.markdown(f'<div class="sample-box">{text}</div>', unsafe_allow_html=True)
                            st.caption(f"Text {i+1}: {len(text)} characters")
                else:
                    st.info("No texts in this cluster")
                
                # Statistics - FIXED VISIBILITY
                if show_stats and cluster_texts:
                    st.markdown("---")
                    st.subheader("📊 Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Text Length Stats:**")
                        lengths = [len(t) for t in cluster_texts[:500]]
                        if lengths:
                            stats_df = pd.DataFrame({
                                'Metric': ['Min', 'Max', 'Average', 'Std Dev'],
                                'Value': [
                                    f"{min(lengths):.0f}",
                                    f"{max(lengths):.0f}",
                                    f"{np.mean(lengths):.1f}",
                                    f"{np.std(lengths):.1f}"
                                ]
                            })
                            # FIX: Clear dataframe display
                            st.dataframe(stats_df, hide_index=True, use_container_width=True)
                    
                    with col2:
                        st.write("**Character Analysis:**")
                        sample_chars = ''.join(cluster_texts[:200])
                        if sample_chars:
                            unique_chars = len(set(sample_chars))
                            total_chars = len(sample_chars)
                            char_data = [
                                ["Total characters", f"{total_chars:,}"],
                                ["Unique characters", unique_chars],
                                ["Diversity", f"{(unique_chars/total_chars*100):.1f}%"]
                            ]
                            char_df = pd.DataFrame(char_data, columns=['Metric', 'Value'])
                            # FIX: Clear dataframe display
                            st.dataframe(char_df, hide_index=True, use_container_width=True)
    
    # TAB 3: ANALYZE TEXT
    with tab3:
        st.title("🎯 Text Analysis")
        
        # Simple text input
        input_text = st.text_area(
            "Enter text to analyze:",
            height=200,
            placeholder="Type or paste text in any Indian language here..."
        )
        
        # Example buttons
        st.write("**Try examples:**")
        col1, col2, col3 = st.columns(3)
        examples = {
            "Hindi": "नमस्ते, आप कैसे हैं? यह हिंदी भाषा है।",
            "Tamil": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்? இது தமிழ் மொழி.",
            "Bengali": "নমস্কার, আপনি কেমন আছেন? এটি বাংলা ভাষা।"
        }
        
        with col1:
            if st.button("Hindi Example", use_container_width=True):
                st.session_state.analysis_text = examples["Hindi"]
        with col2:
            if st.button("Tamil Example", use_container_width=True):
                st.session_state.analysis_text = examples["Tamil"]
        with col3:
            if st.button("Bengali Example", use_container_width=True):
                st.session_state.analysis_text = examples["Bengali"]
        
        # Use session state if example clicked
        if hasattr(st.session_state, 'analysis_text'):
            input_text = st.session_state.analysis_text
        
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if input_text.strip():
                # Text statistics
                text_len = len(input_text)
                word_count = len(input_text.split())
                
                # Script analysis
                st.markdown("---")
                st.subheader("📋 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Characters", text_len)
                with col2:
                    st.metric("Word Count", word_count)
                with col3:
                    indian_chars = sum(1 for c in input_text if '\u0900' <= c <= '\u0D7F')
                    st.metric("Indian Script Chars", indian_chars)
                
                # Script detection
                st.markdown("---")
                st.subheader("🔤 Script Detection")
                
                scripts = {
                    'Devanagari (Hindi/Marathi)': sum(1 for c in input_text if '\u0900' <= c <= '\u097F'),
                    'Bengali/Assamese': sum(1 for c in input_text if '\u0980' <= c <= '\u09FF'),
                    'Tamil': sum(1 for c in input_text if '\u0B80' <= c <= '\u0BFF'),
                    'Telugu': sum(1 for c in input_text if '\u0C00' <= c <= '\u0C7F'),
                    'Gujarati': sum(1 for c in input_text if '\u0A80' <= c <= '\u0AFF'),
                    'Gurmukhi (Punjabi)': sum(1 for c in input_text if '\u0A00' <= c <= '\u0A7F'),
                    'English/Latin': sum(1 for c in input_text if ('a' <= c <= 'z') or ('A' <= c <= 'Z')),
                    'Arabic (Urdu)': sum(1 for c in input_text if '\u0600' <= c <= '\u06FF')
                }
                
                script_data = []
                for script, count in scripts.items():
                    if count > 0:
                        percentage = (count / text_len * 100) if text_len > 0 else 0
                        script_data.append([script, count, f"{percentage:.1f}%"])
                
                if script_data:
                    script_df = pd.DataFrame(script_data, columns=['Script', 'Count', 'Percentage'])
                    st.dataframe(script_df, use_container_width=True, hide_index=True)
                    
                    # Determine dominant script
                    dominant_script = max(scripts.items(), key=lambda x: x[1])[0]
                    st.info(f"**Dominant script:** {dominant_script}")
                
                # Text preview - FIXED VISIBILITY
                st.markdown("---")
                st.subheader("👁️ Text Preview")
                
                # FIX: Use custom box for clear visibility
                preview_text = input_text[:500] + "..." if len(input_text) > 500 else input_text
                st.markdown(f'<div class="sample-box">{preview_text}</div>', unsafe_allow_html=True)
                
            else:
                st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <p>Indian Language Clustering Dashboard • Unsupervised Learning • 🇮🇳</p>
    <p style="font-size: 0.9rem;">K-Means Clustering on Unlabeled Indian Language Text</p>
</div>
""", unsafe_allow_html=True)