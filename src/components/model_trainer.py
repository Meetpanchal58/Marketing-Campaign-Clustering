import os
import pandas as pd
from src.logger.Logging import logging
from src.utils.utils import save_csv, load_csv
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from src.exception.exception import CustomException


@dataclass
class ModelTrainerConfig:
    cleaned_file_path=os.path.join('artifacts','marketing_cleaned.csv')
    clustered_file_path=os.path.join('artifacts','marketing_clustered.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def train_model(self, df):
        logging.info("Model training started")
        
        try:
            # Create a pipeline of the steps
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('pca', PCA(n_components=2)),
                ('kmeans', KMeans(n_clusters=3, random_state=0, init='k-means++'))
            ])

            # Fit the pipeline on the data
            pipeline.fit(df)

            # Extract details accordingly
            pca_components = pipeline.named_steps['pca'].transform(pipeline.named_steps['scaler'].transform(df))
            pca_df = pd.DataFrame(data=pca_components, columns=["PCA1", "PCA2"])
            kmeans_labels = pipeline.named_steps['kmeans'].labels_
            pca_df['Cluster'] = pipeline.named_steps['kmeans'].labels_

            # loading the cleaned dataset for adding clusters
            #data = pd.read_csv(r'C:\Users\meetp\#PYTHON FILES\Customer Segmentation Clustering\artifacts\marketing_cleaned.csv')
            data = load_csv(self.model_trainer_config.cleaned_file_path)

            cluster_df = pd.concat([data, pd.DataFrame({'Cluster': kmeans_labels})], axis=1)
            cluster_df.dropna(inplace=True)
            save_csv(
                file_path=self.model_trainer_config.clustered_file_path,
                data=cluster_df
            )
            #cluster_df.to_csv(r'C:\Users\meetp\#PYTHON FILES\Customer Segmentation Clustering\artifacts\marketing_clustered.csv')

            logging.info("Model training completed")
            return pca_df, kmeans_labels

        except Exception as e:
            logging.exception("An error occurred during model training")
            raise CustomException(e)
