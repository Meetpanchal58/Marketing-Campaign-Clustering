import os
import boto3
from src.logger.Logging import logging
from src.exception.exception import CustomException

class DataIngestion:
    def __init__(self, raw_data_path="artifacts/marketing_campaign.csv"):
        self.raw_data_path = raw_data_path

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Check if the dataset file already exists in the artifacts folder
            dataset_path = os.path.join("artifacts", "marketing_campaign.csv")
            if os.path.exists(dataset_path):
                logging.info("Dataset already exists. Skipping data ingestion.")
                return dataset_path

            # AWS S3 details
            bucket_name = 'meet-db'
            object_key = 'marketing_campaign.csv'
            download_path = os.path.join(os.getcwd(), "artifacts", "marketing_campaign.csv")

            # Download dataset from S3
            s3 = boto3.resource('s3')
            s3.Bucket(bucket_name).download_file(Key=object_key, Filename=download_path)
            logging.info("Dataset downloaded successfully from S3")

            return download_path

        except Exception as e:
            logging.exception("An error occurred during data ingestion")
            raise CustomException(e)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
