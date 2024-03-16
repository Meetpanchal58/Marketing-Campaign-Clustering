import os
import pandas as pd
from dataclasses import dataclass
from src.logger.Logging import logging
from src.utils.utils import load_csv
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataCleaning
from src.components.model_trainer import ModelTrainer
from src.exception.exception import CustomException

@dataclass
class DataTransformationConfig:
    raw_file_path=os.path.join('artifacts','marketing_campaign.csv')

class Training:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def run_training(self):
        try:
            # Data ingestion
            data_ingestion = DataIngestion()
            data_ingestion.initiate_data_ingestion()
            
            #data = pd.read_csv('C:/Users/meetp/#PYTHON FILES/Customer Segmentation Clustering/artifacts/marketing_campaign.csv')
            data = load_csv(self.data_transformation_config.raw_file_path)

            # Data Cleaning
            data_cleaning = DataCleaning()
            df = data_cleaning.clean(data)
 
            # Data transformation
            data_transformation = DataTransformation()
            df = data_transformation.transform_data_pipeline(data)
            
            # Model training
            model_trainer = ModelTrainer()
            pca_df, kmeans_labels = model_trainer.train_model(df)
            
            logging.info("Training completed successfully")
            return pca_df, kmeans_labels
        
        except CustomException as e:
            logging.error(f"An error occurred during training: {e}")
            raise

if __name__ == "__main__":
    training_pipeline = Training()
    training_pipeline.run_training()
