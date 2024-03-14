import os
from src.logger.Logging import logging
from kaggle.api.kaggle_api_extended import KaggleApi
from src.exception.exception import CustomException

class DataIngestion:
    def __init__(self, raw_data_path="artifacts/raw.csv"):
        self.raw_data_path = raw_data_path

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Check if the dataset file already exists in the artifacts folder
            dataset_path = os.path.join("artifacts", "marketing_campaign.csv")
            if os.path.exists(dataset_path):
                logging.info("Dataset already exists. Skipping data ingestion.")
                return dataset_path

            # Authenticate with Kaggle API
            api = KaggleApi()
            api.authenticate()

            # Dataset details
            dataset_name = "imakash3011/customer-personality-analysis"

            # Download dataset
            api.dataset_download_files(dataset_name, path=".", unzip=True)
            logging.info("Dataset downloaded successfully")

            # Create artifacts folder if it doesn't exist in the current directory
            artifacts_path = os.path.join(os.getcwd(), "artifacts")
            os.makedirs(artifacts_path, exist_ok=True)

            # Move dataset to appropriate location
            for file in os.listdir("."):
                if file.endswith(".csv"):
                    os.rename(os.path.join(".", file), os.path.join(artifacts_path, file))
                    logging.info(f"Moved {file} to artifacts folder")

            return os.path.join(artifacts_path, "customer-personality.csv")

        except Exception as e:
            logging.exception("An error occurred during data ingestion")
            raise CustomException(e)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
