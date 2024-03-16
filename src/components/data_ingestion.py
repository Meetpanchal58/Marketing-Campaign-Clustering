import os
import boto3
import pandas as pd
from src.logger.Logging import logging
from src.exception.exception import CustomException

class DataIngestion:
    def __init__(self, raw_data_path="artifacts/marketing_campaign.csv"):
        self.raw_data_path = raw_data_path

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # AWS S3 details
            bucket_name = 'meet-db'
            object_key = 'marketing_campaign.csv'
            # Load dataset from S3
            s3 = boto3.resource('s3')
            obj = s3.Bucket(bucket_name).Object(object_key).get()
            df = pd.read_csv(obj['Body'], delimiter="\t")
            logging.info("Dataset loaded successfully from S3")

            # Save dataset to local "artifacts" folder (optional)
            dataset_path = os.path.join("artifacts", "marketing_campaign.csv")
            df.to_csv(dataset_path, index=False)
            logging.info("Dataset saved to local artifacts folder")

            return dataset_path

        except Exception as e:
            logging.exception("An error occurred during data ingestion")
            raise CustomException(e)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

