from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 
from src.components.model_evaluation import ModelEvaluation
from src.logger.Logging import logging
from src.exception.exception import CustomException


class Complete_Pipline:
    def __init__(self):
        pass

    def run_all_steps(self):
        try:
            # Data ingestion
            data_ingestion = DataIngestion()
            df = data_ingestion.initiate_data_ingestion()

            # Data Cleaning
            data_cleaning = DataCleaning()
            df = data_cleaning.clean(df)
 
            # Data transformation
            data_transformation = DataTransformation()
            df = data_transformation.transform_data_pipeline(df)
            
            # Model training
            model_trainer = ModelTrainer()
            pca_df, kmeans_labels = model_trainer.train_model(df)

            model_evaluation = ModelEvaluation()
            model_evaluation.evaluate_model(pca_df, kmeans_labels)
            
            logging.info("Pipline completed successfully")
        
        except CustomException as e:
            logging.error(f"An error occurred during the process: {e}")
            raise

if __name__ == "__main__":
    training_pipeline = Complete_Pipline()
    training_pipeline.run_all_steps()
