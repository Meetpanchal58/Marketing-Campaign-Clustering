import os
from src.logger.Logging import logging
from dataclasses import dataclass
from src.utils.utils import save_model, load_csv
from sklearn.pipeline import Pipeline
from src.components.data_transformation import Date_Encoding
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Final_Pipline_Config:
    model_file_path=os.path.join('artifacts','kmeans_pipeline.pkl')
    cleaned_file_path = os.path.join('artifacts','marketing_cleaned.csv')


class PipelineBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.Final_Pipline_config=Final_Pipline_Config()

    def fit(self):
        logging.info("Fitting the pipeline...")
        try:
            # Load data
            data = load_csv(self.Final_Pipline_config.cleaned_file_path)

            pipeline = Pipeline([
                ('date_transformer', Date_Encoding()), 
                ('preprocessor', ColumnTransformer([
                    ('categorical', OneHotEncoder(), ['Education', 'Marital_Status']),
                ], remainder='passthrough')),
                ('scaler', RobustScaler()),
                ('pca', PCA(n_components=2)),
                ('kmeans', KMeans(n_clusters=3, random_state=0, init='k-means++'))
            ])
            pipeline.fit(data)
            
            save_model(
                file_path=self.Final_Pipline_config.model_file_path,
                obj=pipeline
            )
            #with open('C:/Users/meetp/#PYTHON FILES/Customer Segmentation Clustering/artifacts/kmeans_pipeline.pkl', 'wb') as f:
                #pickle.dump(pipeline, f)

            logging.info("Pipeline fitting completed.")
            return pipeline

        except Exception as e:
            logging.error("An error occurred during pipeline fitting.")
            raise e

if __name__ == "__main__":
    # Instantiate and fit pipeline
    pipeline_builder = PipelineBuilder()
    pipeline_builder.fit()