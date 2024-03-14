import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.logger.Logging import logging
import dagshub
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_figure
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from src.exception.exception import CustomException

class ModelEvaluation:
    def __init__(self):
        pass

    def evaluate_model(self, pca_df, kmeans_labels):
        logging.info("Model evaluation started")
        
        try:
            dagshub.init(repo_owner='Meetpanchal58', repo_name='Customer-Segmentation-Clustering', mlflow=True)
            
            remover_server_uri = "https://dagshub.com/Meetpanchal58/Customer-Segmentation-Clustering.mlflow"
            mlflow.set_tracking_uri(remover_server_uri)

            # Start MLflow run
            mlflow.start_run(run_name="Model Evaluation")

            # Plot clustering results
            plt.figure(figsize=(8, 8))
            ax = sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=pca_df, palette=['red', 'green', 'blue'])
            plt.title("Clustering using K-Means Algorithm")
            plt.show()
            
            mlflow.log_figure(ax.figure, 'clustering_plot.png')

            # Evaluate clustering
            silhouette = silhouette_score(pca_df, kmeans_labels)
            calinski_harabasz = calinski_harabasz_score(pca_df, kmeans_labels)
            davies_bouldin = davies_bouldin_score(pca_df, kmeans_labels)

            # Log clustering evaluation metricc
            log_metric("KMeans Silhouette Score", silhouette)
            log_metric("KMeans Calinski Harabasz Score", calinski_harabasz)
            log_metric("KMeans Davies Bouldin Score", davies_bouldin)

            # Load data with clusters
            cluster_df = pd.read_csv('C:/Users/meetp/Downloads/!PYTHON FILES/MLops-Project/artifacts/marketing_encoded.csv')
            cluster_df['Cluster'] = kmeans_labels

            # Split data for classification
            X = cluster_df.drop(labels=["Cluster"], axis=1)
            Y = cluster_df["Cluster"]
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

            # Classification models
            models = {
                "RandomForest": RandomForestClassifier(criterion="entropy"),
                "DecisionTree": DecisionTreeClassifier(criterion="entropy")
            }

            for name, model in models.items():
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                accuracy = accuracy_score(Y_test, Y_pred)
                cm = confusion_matrix(Y_test, Y_pred)
                classification_rep = classification_report(Y_test, Y_pred)

                # Log model metrics
                log_metric(f"{name} Accuracy", accuracy)

                log_param(f"{name} Confusion Matrix", cm.tolist())
                log_param(f"{name} Classification Report", classification_rep)
    
            logging.info("Model evaluation completed")
        
        except Exception as e:
            logging.exception("An error occurred during model evaluation")
            raise CustomException(e)
        
        finally:
            # End MLflow run
            mlflow.end_run()