from src.logger.Logging import logging
from src.ModelTesting.training import Training
from src.components.model_evaluation import ModelEvaluation
from src.exception.exception import CustomException


class Prediction:
    def __init__(self):
        pass

    def run_prediction(self):
        try:
            # Run the training pipeline to get the necessary data
            training_pipeline = Training()
            pca_df, kmeans_labels = training_pipeline.run_training()
            
            # Model evaluation
            model_evaluation = ModelEvaluation()
            model_evaluation.evaluate_model(pca_df, kmeans_labels)
            
            logging.info("Prediction completed successfully")
        
        except CustomException as e:
            logging.error(f"An error occurred during prediction: {e}")
            raise

if __name__ == "__main__":
    prediction_pipeline = Prediction()
    prediction_pipeline.run_prediction()
