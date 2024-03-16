from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
import pendulum
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 
from src.components.model_evaluation import ModelEvaluation
from src.pipline.Pipline_pkl import PipelineBuilder

with DAG(
    "Clustering_Complete_pipeline",
    default_args={"retries": 2},
    description="this is Complete ML Pipeline",
    schedule="@weekly",# here you can test based on hour or mints but make sure here you container is up and running
    start_date=pendulum.datetime(2024, 2, 17, tz="UTC"),
    catchup=False,
    tags=["unsupervised","machine learning","clustering"],
) as dag:
    
    dag.doc_md = __doc__


def Run_Data_ingestion_pipline(**kwargs):
        data_ingestion = DataIngestion()
        df = data_ingestion.initiate_data_ingestion()
        kwargs['ti'].xcom_push(key='raw_df', value=df)


def Run_Data_cleaning_pipline(**kwargs):
        df = kwargs['ti'].xcom_pull(key='raw_df')
        data_cleaning = DataCleaning()
        df = data_cleaning.clean(df)
        kwargs['ti'].xcom_push(key='cleaned_df', value=df)
     

def Run_Data_transformation_pipline(**kwargs):
        df = kwargs['ti'].xcom_pull(key='cleaned_df')
        data_transformation = DataTransformation()
        df = data_transformation.transform_data_pipeline(df)
        kwargs['ti'].xcom_push(key='encoded_df', value=df)
     

def Run_Model_trainer_pipline(**kwargs):
        df = kwargs['ti'].xcom_pull(key='encoded_df')
        model_trainer = ModelTrainer()
        pca_df, kmeans_labels = model_trainer.train_model(df)
        #pca_df = pca_df.to_dict(orient='records')
        #kmeans_labels = kmeans_labels.tolist()
        kwargs['ti'].xcom_push(key='pca_df', value=pca_df)
        kwargs['ti'].xcom_push(key='kmeans_labels', value=kmeans_labels)
     
     
def Run_Model_evaluation_pipline(**kwargs):
        pca_df = kwargs['ti'].xcom_pull(key='pca_df')
        kmeans_labels = kwargs['ti'].xcom_pull(key='kmeans_labels')
        model_evaluation = ModelEvaluation()
        model_evaluation.evaluate_model(pca_df, kmeans_labels)


def Run_Pipline_pkl_pipeline():
    pipeline_builder = PipelineBuilder()
    pipeline_builder.fit()


def push_to_s3():
        subprocess.run(["dvc", "commit"])
        subprocess.run(["dvc", "push", "-r", "MyStorage"])
      

data_ingestion_task = PythonOperator(
    task_id='run_data_ingestion_pipeline',
    python_callable=Run_Data_ingestion_pipline,
    provide_context=True,  
    dag=dag,
)

data_cleaning_task = PythonOperator(
    task_id='run_Data_cleaning_pipeline',
    python_callable=Run_Data_cleaning_pipline,
    provide_context=True,  
    dag=dag,
)

data_transformation_task = PythonOperator(
    task_id='run_Data_transformation_pipeline',
    python_callable=Run_Data_transformation_pipline,
    provide_context=True,  
    dag=dag,
)

model_trainer_task = PythonOperator(
    task_id='run_model_trainer_pipeline',
    python_callable=Run_Model_trainer_pipline,
    provide_context=True,  
    dag=dag,
)

model_evaluation_task = PythonOperator(
    task_id='run_model_evaluation_pipeline',
    python_callable=Run_Model_evaluation_pipline,
    provide_context=True,  
    dag=dag,
)

Pipline_pkl_task = PythonOperator(
    task_id='run_final_pkl_pipeline',
    python_callable=Run_Pipline_pkl_pipeline,
    provide_context=True, 
    dag=dag,
)

push_to_s3_task = PythonOperator(
    task_id='push_to_s3',
    python_callable=push_to_s3,
    dag=dag,
)


# Set task dependencies
data_ingestion_task >> data_cleaning_task >> data_transformation_task >> model_trainer_task >> model_evaluation_task >> Pipline_pkl_task >> push_to_s3_task
