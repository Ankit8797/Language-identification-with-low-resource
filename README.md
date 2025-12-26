# Language-identification-with-low-resource
A Machine learning model for recognition of indian languages with low resources
•	India is a linguistically diverse country with hundreds of languages written across multiple scripts such as Devanagari, Tamil, Bengali, Telugu, and others. With the rapid growth of digital content on social media platforms, news portals, and user-generated text, the automatic identification and analysis of Indian languages have become an important problem in the fields of Natural Language Processing (NLP) and Machine Learning.

•	Traditional language identification systems often rely on large amounts of labelled data. However, for many Indian languages, labelled datasets are limited, inconsistent, or difficult to obtain. This creates a strong motivation to explore unsupervised and language-aware approaches that can discover patterns directly from raw text without prior labelling.


•	This project focuses on the unsupervised clustering of Indian language text data using a combination of statistical text features, character-level analysis, and script-based heuristics. The system extracts multiple feature representations such as TF-IDF features, character n-grams, and language-specific Unicode patterns to capture the structural and linguistic properties of different Indian languages. Dimensionality reduction techniques are applied to improve clustering efficiency, followed by clustering algorithms such as K-Means and ensemble-based methods to group text samples into linguistically coherent clusters.

•	In addition to clustering, the project incorporates internal evaluation metrics and visualization techniques to assess cluster quality and interpretability. A Streamlit-based interactive dashboard is developed to enable intuitive exploration of clustering results, making the system suitable for both analysis and demonstration purposes.
