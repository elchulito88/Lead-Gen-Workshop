import domino
import os
import time

# Import mlflow & initiate client
import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.projects

 
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