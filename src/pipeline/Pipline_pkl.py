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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Final_Pipline_Config:
    model_file_path=os.path.join('artifacts','kmeans_pipeline.pkl')
    cleaned_file_path = os.path.join('artifacts','marketing_cleaned.csv')
    clustered_file_path = os.path.join('artifacts','marketing_clustered.csv')
    model2_file_path = os.path.join('artifacts','GradientBoosting_pipeline.pkl')


class PipelineBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.Final_Pipline_config=Final_Pipline_Config()

    def fit(self):
        try:
            logging.info("Fitting the Kmeans pipeline ...")

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

            logging.info("Kmeans Pipeline fitting completed.")


            logging.info("Fitting the Kmeans pipeline ...")

            data2 = load_csv(self.Final_Pipline_config.clustered_file_path)

            pipeline2 = Pipeline([
                ('preprocessor', ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(), ['Education', 'Marital_Status']),
                        ('date', Date_Encoding(), ['Dt_Customer'])
                    ],
                    remainder='passthrough'
                )),
                ('classifier', GradientBoostingClassifier())
            ])

            # Split data into features and target
            X = data2.drop(labels=["Cluster"], axis=1)
            y = data2["Cluster"]

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

            # Fit the pipeline
            pipeline2.fit(X_train, y_train)

            # Save the pipeline
            save_model(
                file_path=self.Final_Pipline_config.model2_file_path,
                obj=pipeline2
            )

            logging.info("GradientBoosting Pipeline fitting completed.")

            return pipeline, pipeline2

        except Exception as e:
            logging.error("An error occurred during pipeline fitting.")
            raise e

if __name__ == "__main__":
    # Instantiate and fit pipeline
    pipeline_builder = PipelineBuilder()
    pipeline_builder.fit()