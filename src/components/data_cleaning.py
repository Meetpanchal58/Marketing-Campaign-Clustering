import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger.Logging import logging
from src.utils.utils import save_csv
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import zscore
from src.exception.exception import CustomException


@dataclass
class DataTransformationConfig:
    cleaned_file_path=os.path.join('artifacts','marketing_cleaned.csv')


class Data_Cleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info("Data cleaning started")

        try:
            # Handle missing values in the 'Income' column
            X['Income'] = X['Income'].fillna(X['Income'].mean())

            # Create Age column
            current_year = pd.Timestamp.now().year
            X['Age'] = current_year - X['Year_Birth']

            # Renaming Marital_Status Labels
            X["Marital_Status"] = X["Marital_Status"].replace({"Married": "Partner", "Together": "Partner",
                                                               "Absurd": "Single", "YOLO": "Single",
                                                               "Widow": "Single", "Divorced": "Single",
                                                               "Alone": "Single"})

            # Renaming Education Labels
            X['Education'] = X['Education'].replace({"Basic": "UnderGraduate", "2n Cycle": "UnderGraduate"})

            X['Dt_Customer'] = pd.to_datetime(X['Dt_Customer'], format='%d-%m-%Y')

            # Drop unnecessary columns
            X.drop(['ID', 'Z_CostContact', 'Year_Birth', 'Z_Revenue', 'Complain', 'NumWebVisitsMonth'], axis=1, inplace=True)

            # Remove outliers
            X = X[(np.abs(zscore(X['Age'])) < 3) & (np.abs(zscore(X['Income'])) < 3)]
            X.reset_index(drop=True, inplace=True)
            #X.to_csv(r'C:\Users\meetp\#PYTHON FILES\Customer Segmentation Clustering\artifacts\marketing_cleaned.csv', index=False)
            save_csv(
                file_path=self.data_transformation_config.cleaned_file_path,
                data=X
            )

            logging.info("Data cleaning completed")
            return X
        
        except Exception as e:
            logging.exception("An error occurred during data cleaning")
            raise CustomException(e)


class DataCleaning:
    def __init__(self):
        pass

    def clean(self, df):
        try:
            pipeline = Pipeline([
            ('data_cleaning', Data_Cleaning())
            ])

            logging.info("Data cleaning pipeline is started")
            cleaned_df = pipeline.fit_transform(df)
            logging.info("Data cleaning pipeline is completed")

            return cleaned_df

        except Exception as e:
            logging.exception("An error occurred during data cleaning pipeline")
            raise CustomException(e)