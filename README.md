# Customer-Churn-Prediction-Using-ZenML--MLOPS


Problem statement: For a given customer's historical data, we are asked to predict whether a customer will stop using a company's product or not. We will be using the Telco Customer Churn dataset for building an end to end production-grade machine learning system that can predict whether the customer will stay loyal or not. The dataset has 20 input features and a target variable for 7043 customers.

When someone leaves a company and when that customer stops paying a business for its services or products, we call that 'churn'. We can calculate a churn rate for a company by dividing the number of customers who churned by the total number of customers and then multiplying that number by 100 to reach a percentage value. If you want to learn more about customer churn, you can read this Wikipedia article.

To achieve this in a real-world scenario, we will be using ZenML to build a production-ready pipeline that predicts whether a customer will churn or not ahead of time. The purpose of this repository is to demonstrate how ZenML empowers your business to build and deploy machine learning pipelines in a multitude of ways:

- By offering you a framework or template to develop within.
- By integrating with popular and useful tools like Kubeflow, Seldon Core, facets, and more.
- By allowing you to build and deploy your machine learning pipelines easily using a modern MLOps framework.


## Dataset
https://www.kaggle.com/datasets/blastchar/telco-customer-churn?datasetId=13996&sortBy=voteCount



## Python Requirements
Let's jump into the Python packages you need. Within the Python environment of your choice, run:
```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd zenml-projects/customer-satisfaction
pip install -r requirements.txt
```
Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to launch the ZenML Server and Dashboard locally, but first you must install the optional dependencies for the ZenML server:
```bash
pip install zenml["server"]
zenml up
```

If you are running the run_deployment.py script, you will also need to install some integrations using ZenML:
```bash
zenml integration install mlflow -y

## The Solution
In order to build a real-world workflow for predicting the customer satisfaction score for the next order or purchase (which will help make better decisions), it is not enough to just train the model once.

Instead, we are building an end-to-end pipeline for continuously predicting and deploying the machine learning model, alongside a data application that utilizes the latest deployed model for the business to consume.

This pipeline can be deployed to the cloud, scale up according to our needs, and ensure that we track the parameters and data that flow through every pipeline that runs. It includes raw data input, features, results, the machine learning model and model parameters, and prediction outputs. ZenML helps us to build such a pipeline in a simple, yet powerful, way.

In this Project, we give special consideration to the MLflow integration of ZenML. In particular, we utilize MLflow tracking to track our metrics and parameters, and MLflow deployment to deploy our model. We also use Streamlit to showcase how this model will be used in a real-world setting.
```

## Training Pipeline
Our standard training pipeline consists of several steps:

- ingest_data: This step will ingest the data and create a DataFrame.
- clean_data: This step will clean the data and remove the unwanted columns.
- train_model: This step will train the model and save the model using MLflow autologging.
- evaluation: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.


## Deployment Pipeline
We have another pipeline, the deployment_pipeline.py, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria. The criteria that we have chosen is a configurable threshold on the MSE of the training. The first four steps of the pipeline are the same as above, but we have added the following additional ones:

- deployment_trigger: The step checks whether the newly trained model meets the criteria set for deployment.
- model_deployer: This step deploys the model as a service using MLflow (if deployment criteria is met).
In the deployment pipeline, ZenML's MLflow tracking integration is used for logging the hyperparameter values and the trained model itself and the model evaluation metrics -- as MLflow experiment tracking artifacts -- into the local MLflow backend. This pipeline also launches a local MLflow deployment server to serve the latest MLflow model if its accuracy is above a configured threshold.

The MLflow deployment server runs locally as a daemon process that will continue to run in the background after the example execution is complete. When a new pipeline is run which produces a model that passes the accuracy threshold validation, the pipeline automatically updates the currently running MLflow deployment server to serve the new model instead of the old one.

To round it off, we deploy a Streamlit application that consumes the latest model service asynchronously from the pipeline logic. This can be done easily with ZenML within the Streamlit code:
```bash
service = prediction_service_loader(
   pipeline_name="continuous_deployment_pipeline",
   pipeline_step_name="mlflow_model_deployer_step",
   running=False,
)
...
service.predict(...)  # Predict on incoming data from the application
```
## Requirements
## Experiments
## Contributions





