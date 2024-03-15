from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
import pendulum
from src.pipline.training import Training
from src.pipline.evaluation import Prediction
from src.pipline.final_model import PipelineBuilder

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


# Define tasks for training pipeline
def run_training_pipeline(**kwargs):
    training_pipeline = Training()
    pca_df, kmeans_labels = training_pipeline.run_training()
    kwargs['ti'].xcom_push(key='pca_df', value=pca_df)
    kwargs['ti'].xcom_push(key='kmeans_labels', value=kmeans_labels)

train_task = PythonOperator(
    task_id='run_training_pipeline',
    python_callable=run_training_pipeline,
    provide_context=True,  # Pass context to the Python function
    dag=dag,
)

# Define tasks for prediction pipeline
def run_prediction_pipeline(**kwargs):
    pca_df = kwargs['ti'].xcom_pull(key='pca_df')
    kmeans_labels = kwargs['ti'].xcom_pull(key='kmeans_labels')
    prediction_pipeline = Prediction()
    prediction_pipeline.run_prediction(pca_df, kmeans_labels)

predict_task = PythonOperator(
    task_id='run_prediction_pipeline',
    python_callable=run_prediction_pipeline,
    provide_context=True,  # Pass context to the Python function
    dag=dag,
)

# Define task for final model pipeline
def run_final_model_pipeline():
    pipeline_builder = PipelineBuilder()
    pipeline_builder.fit()


final_model_task = PythonOperator(
    task_id='run_final_model_pipeline',
    python_callable=run_final_model_pipeline,
    provide_context=True,  # Pass context to the Python function
    dag=dag,
)

# Define function to push data to S3 using DVC
def push_to_s3():
        subprocess.run(["dvc", "commit"])
        subprocess.run(["dvc", "push", "-r", "MyStorage"])

# Add push_to_s3 task to the DAG
push_to_s3_task = PythonOperator(
    task_id='push_to_s3',
    python_callable=push_to_s3,
    dag=dag,
)


# Set task dependencies
train_task >> predict_task >> final_model_task >> push_to_s3_task
