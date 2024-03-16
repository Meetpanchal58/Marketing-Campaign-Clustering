import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger.Logging import logging
from src.utils.utils import save_csv
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from scipy.stats import zscore
from src.exception.exception import CustomException


@dataclass
class DataTransformationConfig:
    encoded_file_path=os.path.join('artifacts','marketing_encoded.csv')


class Date_Encoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy['Dt_Customer'] = pd.to_datetime(X_copy['Dt_Customer'])
        X_copy['Day_of_Joining'] = X_copy['Dt_Customer'].dt.day
        X_copy['Month_of_Joining'] = X_copy['Dt_Customer'].dt.month
        X_copy['Year_of_Joining'] = X_copy['Dt_Customer'].dt.year
        X_copy.drop(['Dt_Customer'], axis=1, inplace=True)

        return X_copy


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def transform_data_pipeline(self, df):
        try:
            pipeline = Pipeline([
                ('date_transformer', Date_Encoding()),  # Convert datetime to numeric
                ('preprocessor', ColumnTransformer([
                    ('categorical', OneHotEncoder(), ['Education', 'Marital_Status']),  # One-hot encoding for categorical columns
                ], remainder='passthrough'))
            ])

            logging.info("Data transformation pipeline is started")
            transformed_data = pipeline.fit_transform(df)
            logging.info("Data transformation pipeline is completed")

            transformed_data = pd.DataFrame(transformed_data)
            #transformed_data.to_csv(r'C:\Users\meetp\#PYTHON FILES\Customer Segmentation Clustering\artifacts\marketing_encoded.csv',index=False, header=True)
            save_csv(
                file_path=self.data_transformation_config.encoded_file_path,
                data=transformed_data
            )

            return transformed_data

        except Exception as e:
            logging.exception("An error occurred during data transformation pipeline")
            raise CustomException(e)
