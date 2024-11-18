import h2o
from h2o.automl import H2OAutoML
import json
import pickle 
import pandas as pd
import numpy as np
import random
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Import mlflow
import mlflow
import mlflow.h2o
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.models.signature import infer_signature

#Set train test split to 70
n = 70
 
#read in data then split into train and test
 
path = str('/mnt/data/mlops-best-practices/credit_card_default.csv')
data = pd.read_csv(path)
print('Read in {} rows of data'.format(data.shape[0]))
  
# Get data set up for model training and evaluation
 
# Drop NA rows
data = data.dropna(how='any',axis=0)

# Split df into inputs and target

# Downsample data for speed

columns_to_model = ['PAY_0', 'PAY_2', 'PAY_4', 'LIMIT_BAL', 'PAY_3', 'BILL_AMT1', 'DEFAULT']

data = data.sample(n=1000, random_state=42, axis=0)

data = data[columns_to_model]

train = data[0:round(len(data)*n/100)]
test = data[train.shape[0]:]
 
print('H2O version -{}'.format(h2o.__version__))
 
#initailize local h2o
h2o.init()
 
#Convert data to h2o frames
hTrain = h2o.H2OFrame(train)
hTest = h2o.H2OFrame(test)
 
# Identify predictors and response
x = hTrain.columns
y = "DEFAULT"
x.remove(y)
 
# Isolate target variable
hTrain[y] = hTrain[y].asfactor()
hTest[y] = hTest[y].asfactor()
 
# Run AutoML for 5 base models (limited to 1 min max runtime)
print('Training autoML model...')
aml = H2OAutoML(max_models=10, max_runtime_secs=30, sort_metric="auc")
aml.train(x=x, y=y, training_frame=hTrain)
 
print('Evaluating model on validation data...')
best_model = aml.get_best_model(criterion = 'logloss') 
preds = best_model.predict(hTest)

#View performance metrics and save them to domino stats!
auc = np.round(best_model.auc(xval=True), 4)
logloss = np.round(best_model.logloss(xval=True), 4)
f1_score = np.round(best_model.F1()[0][0], 4)
precision_score = np.round(best_model.precision()[0][0], 4)
recall_score = np.round(best_model.recall()[0][0], 4)


print("AUC: ", auc)
print("logloss: ", logloss)

# Write top model performance to dominostats for Jobs tracker
with open('dominostats.json', 'w') as f:
    f.write(json.dumps({"AUC": auc,
                       "logloss": logloss}))
 
# Write results to dataframe for viz    
results = pd.DataFrame({'Actuals':test['DEFAULT'], 'Predictions': preds.as_data_frame()['predict']})
 
print('Creating visualizations...')

# Permutation Importance Plot
PI_plot = best_model.permutation_importance_plot(hTest).figure()
PI_plot.savefig('/mnt/artifacts/h2o_PI_plot.png')

# SHAP Summary Plot
shap_plot = best_model.shap_summary_plot(hTest).figure()
shap_plot.savefig('/mnt/artifacts/1_h2o_SHAP_Summary.png')

#Saving trained model to serialized pickle object 
aml_path = h2o.save_model(best_model, path ='/mnt/code/models')

# Log results in Experiment Tracker
experiment = mlflow.set_experiment(experiment_name=os.environ.get('DOMINO_PROJECT_NAME') + " " + os.environ.get('DOMINO_STARTING_USERNAME'))

# Unique name for Experiment Run
now = datetime.now()
dt_string = now.strftime("%m/%d/%Y")

run_name = 'AutoML Run ' + dt_string

with mlflow.start_run(run_name=run_name) as run:
    
    print('Logging H2O AutoML models to Experiment Tracker')
    
    # Set MLFlow tag to differenciate the model approaches
    mlflow.set_tag("Model_Type", "H2O")

    # Log Model in the Model Registry
    print('Logging top h2O model to Model Catalog')
 
    model_name = 'H2O_AML_model_{}'.format(dt_string)
    
    # Specify model signature for scoring
    signature = infer_signature(train, test)
    print(f"model signature: {signature}")
    
    # Log top model in the Experiment Tracker
    print('Logging Best AutoML model to Experiment Tracker')
    
    # Top Model Parameters
        
    mlflow.log_param("model_type", "H2O")
    mlflow.log_param("model_id", best_model.actual_params['model_id'])

    # Top Model Metrics
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("logloss", logloss)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("precision_score", precision_score)
    mlflow.log_metric("recall_score", recall_score)

    # Top Model Artifacts
    mlflow.log_artifact('/mnt/artifacts/h2o_PI_plot.png')
    mlflow.log_artifact('/mnt/artifacts/1_h2o_SHAP_Summary.png')
    mlflow.log_artifact(aml_path)

    # Top Model
    mlflow.h2o.log_model(best_model,
                         artifact_path="h2o_model",
                         signature=signature,
                         )
    
    # You can register your model from here, or from the Domino Experiments Tracker
    
    # client = mlflow.tracking.MlflowClient()
    # client.create_registered_model(model_name)
    
    # Log other top AutoML models as child runs
    
    model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
    
    rank = 1
    
    for lb_model in model_ids[1:5]:
                
        child_run = 'AML model rank: {}'.format(str(rank))
        
        m_id = model_ids[rank]        
        model = h2o.get_model(m_id)
        
        with mlflow.start_run(run_name=child_run, nested=True) as child_run:
            
            mlflow.set_tag("Model_Type", "h2o")
            
            auc = np.round(model.auc(xval=True), 4)
            logloss = np.round(model.logloss(xval=True), 4)
            f1_score = np.round(model.F1()[0][0], 4)
            precision_score = np.round(model.precision()[0][0], 4)
            recall_score = np.round(model.recall()[0][0], 4)
            
            # Model Parameters
            mlflow.log_param("model_type", "H2O")
            mlflow.log_param("model_id", model.actual_params['model_id'])

            # Model Metrics
            mlflow.log_metric("AUC", auc)
            mlflow.log_metric("logloss", logloss)
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            
            mlflow.h2o.log_model(model,
                                 artifact_path="h2o_model",
                                 signature=signature,
                                 )
            
        rank += 1
        

mlflow.end_run()
 
print('Script complete!')