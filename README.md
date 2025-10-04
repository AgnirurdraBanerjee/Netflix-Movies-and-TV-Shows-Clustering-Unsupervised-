🎬 Netflix Content Clustering using Unsupervised Machine Learning
📖 Project Overview
The Netflix content library is a massive and evolving collection of movies and TV shows across diverse genres, languages, and themes. Understanding this vast catalog manually is nearly impossible.
This project uses unsupervised machine learning (clustering) to automatically segment Netflix content into meaningful groups based on genre, description, cast, and director. The goal is to reveal hidden patterns, thematic structures, and relationships within the catalog to support data-driven decision-making for content acquisition, marketing, and personalization.


🎯 Objectives

1. Apply clustering algorithms to segment Netflix content into similar groups.
2. Identify thematic and stylistic patterns within the content catalog.
3. Enable better decision-making for:
   a. Content acquisition and curation
   b. Personalized recommendations
   c. Targeted marketing strategies


🧠 Methodology

1. Data Preprocessing

Text cleaning, tokenization, and feature extraction using TF-IDF Vectorization
Dimensionality reduction with Truncated SVD
Feature scaling using StandardScaler

2. Modeling

Model 1: K-Means Clustering
Elbow Method and Silhouette Score for optimal cluster selection
Model 2: Agglomerative Hierarchical Clustering
Dendrogram visualization and Silhouette analysis
Hyperparameter tuning using manual grid search with silhouette evaluation

3. Evaluation
Inertia (WCSS) and Silhouette Score as evaluation metrics
Visualizations of cluster distribution and performance charts


📊 Technologies Used

Programming Language: Python

Libraries & Tools:

scikit-learn – Clustering models, preprocessing, and metrics
pandas, numpy – Data manipulation
matplotlib, seaborn – Visualization
nltk / re – Text preprocessing
pickle / joblib – Model persistence


💾 Model Saving & Loading

The trained components are serialized for reuse:
kmeans_model.pkl – Optimized K-Means model
tfidf_vectorizer.pkl – TF-IDF feature extractor
svd_model.pkl – Dimensionality reduction model
scaler.pkl – Feature scaler
These components are reloaded to perform predictions on new, unseen data without retraining.


🧩 Results

The optimized K-Means model produced well-separated clusters with improved silhouette scores.
Clusters represent distinct thematic groups such as comedy specials, family animations, thrillers, etc.
The insights enable Netflix stakeholders to understand the content landscape more effectively.


🚀 Future Enhancements

Incorporate NLP embeddings (BERT, Sentence Transformers) for richer semantic understanding.
Automate genre labeling using topic modeling.
Build an interactive dashboard for visual cluster exploration.
Integrate with a recommendation system pipeline.


🧑‍💻 Author

Project by: AGNIMITRA BANERJEE
Domain: Machine Learning / Data Science
Focus Area: Unsupervised Learning, NLP, and Content Analytics
