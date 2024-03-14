import pandas as pd
import numpy as np
from src.logger.Logging import logging
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from scipy.stats import zscore
from src.exception.exception import CustomException


class Date_Encoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy['Day_of_Joining'] = X_copy['Dt_Customer'].dt.day
        X_copy['Month_of_Joining'] = X_copy['Dt_Customer'].dt.month
        X_copy['Year_of_Joining'] = X_copy['Dt_Customer'].dt.year
        X_copy.drop(['Dt_Customer'], axis=1, inplace=True)

        return X_copy


class Data_Cleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

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
            X.to_csv(r'C:\Users\meetp\#PYTHON FILES\Customer Segmentation Clustering\artifacts\marketing_cleaned.csv', index=False)
            #X.to_excel(r'C:\Users\meetp\Downloads\!PYTHON FILES\MLops-Project\artifacts\marketing_cleaned.xlsx', index=False)

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
    

class DataTransformation:
    def __init__(self):
        pass

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
            transformed_data.to_csv(r'C:\Users\meetp\#PYTHON FILES\Customer Segmentation Clustering\artifacts\marketing_encoded.csv',index=False, header=True)
            return transformed_data

        except Exception as e:
            logging.exception("An error occurred during data transformation pipeline")
            raise CustomException(e)
