import domino
import os
import time

# Import mlflow & initiate client
import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.projects

print('Initializing MLFlow Client')

# Initiate MLFlow client
client = mlflow.tracking.MlflowClient()

# Verify MLFLow URI
print('MLFLOW_TRACKING_URI: ' + os.environ['MLFLOW_TRACKING_URI'])

experiment_name = os.environ.get('DOMINO_PROJECT_NAME') + " " + os.environ.get('DOMINO_STARTING_USERNAME')
experiment = mlflow.set_experiment(experiment_name=experiment_name) #changeit

print('Experiment "{}" created, Experiment ID: {}'.format(experiment_name, experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
 
print('Initializing Domino Project for API calls')

# Create the output image directory in a Domino Dataset, if it doesn't already exist
path = '/domino/datasets/local/{}/visualizations'.format(os.environ.get('DOMINO_PROJECT_NAME'))

if not os.path.exists(path):
    os.makedirs(path)
 
#initialize Domino Project
domino_project =domino.Domino(project = str(os.environ.get('DOMINO_PROJECT_OWNER')+'/'+os.environ.get('DOMINO_PROJECT_NAME')),
                              domino_token_file=os.environ.get('DOMINO_TOKEN_FILE'))
 
print('Kicking off sklearn logisitc regression model training')
domino_project.job_start(command='scripts/sklearn_log_reg_train.py')
 
print('Kicking off h2o model training')
domino_project.job_start(command='scripts/h2o_model_train.py')
  
print('Kicking off sklearn random forest model training')
domino_project.job_start(command='scripts/sklearn_RF_train.py')

print('Kicking off XGBoost model training')
domino_project.job_start(command='scripts/xgb_model_train.py')



print('Done!')