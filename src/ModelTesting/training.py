import pandas as pd
from src.logger.Logging import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation, DataCleaning
from src.components.model_trainer import ModelTrainer
from src.exception.exception import CustomException

class Training:
    def __init__(self):
        pass

    def run_training(self):
        try:
            # Data ingestion
            data_ingestion = DataIngestion()
            data_ingestion.initiate_data_ingestion()
            
            data = pd.read_csv('C:/Users/meetp/#PYTHON FILES/Customer Segmentation Clustering/artifacts/marketing_campaign.csv')

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
