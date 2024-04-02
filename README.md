# Marketing Campaign Clustering

### Overview
- This project focuses on clustering customers based on their personal details, product spending, platform purchases, and responses to promotional campaigns. The goal is to identify target customers for various products and campaigns, optimizing resource allocation and reducing marketing costs.

### Dataset Information
- The dataset comprises 2200+ rows and 23 columns, containing customer information and responses to marketing campaigns.

### Problem Statement
- Identifying target customers for products and promotional campaigns is challenging, leading to inefficient resource allocation and increased marketing costs.

### Solution
- We implemented Kmeans clustering, achieving an accuracy exceeding 95%, to categorize customers into 3 clusters based on their preferences, purchase behavior, and response rates. This profiling enables targeted marketing efforts, optimizing resource allocation and reducing unnecessary expenses.

### Tech Stack
- Scikit-learn
- Kmeans Clustering
- Imblearn Pipeline
- DVC (Data Version Control)
- MLFlow-Dagshub (Experiment Tracking)
- Docker (Product Containerization )
- Airflow (Pipeline Orchestation )
- Github Actions CI/CD
- AWS (Amazon Web Services)
- ECR (Elastic Container Registry)
- EC2 (Elastic Compute Cloud)
- Streamlit Cloud

export MLFLOW_TRACKING_URI=https://dagshub.com/Meetpanchal58/Marketing-Campaign-Clustering.mlflow
python src/pipeline/full_pipline.py

deployment - https://marketing-campaign-clustering.streamlit.app/
