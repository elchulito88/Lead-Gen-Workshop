import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, f1_score, precision_score, recall_score, precision_recall_curve, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime

# Import mlflow
import mlflow
import mlflow.sklearn
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.models.signature import infer_signature

def create_visuals(model, param):
    
    print('Creating visualizations for param {}'.format(str(param)))

    # Add visualizations and save for inspection
    # RocCurveDisplay.from_estimator(log_reg, X_test, y_test, name='Logistic Regression AUC Curve')
    roc_curve_display = sklearn.metrics.plot_roc_curve(model, X_test, y_test)
    fig = roc_curve_display.figure_
    plt.savefig('/mnt/artifacts/log_reg_ROC_Curve_C={}.png'), str(param)

    y_pred = log_reg.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    cm_display = ConfusionMatrixDisplay(cm).plot()

    plt.savefig('/mnt/artifacts/log_reg_confusion_matrix_C={}.png'), str(param)

    prec, recall, _ = precision_recall_curve(y_test, y_pred,
                                             pos_label=log_reg.classes_[1])

    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

    plt.savefig('/mnt/artifacts/log_reg_precision_recall_C={}.png'), str(param)

    
    mlflow.log_artifact('/mnt/artifacts/log_reg_ROC_Curve_C={}.png'), str(param)
    mlflow.log_artifact('/mnt/artifacts/log_reg_confusion_matrix_C={}.png'), str(param)
    mlflow.log_artifact('/mnt/artifacts/log_reg_precision_recall_C={}.png'), str(param)  

#read in data then split into train and test
 
path = str('/mnt/data/mlops-best-practices/credit_card_default.csv')
df = pd.read_csv(path)
print('Read in {} rows of data'.format(df.shape[0]))
  
#Get data set up for model training and evaluation
 
#Drop NA rows
df = df.dropna(how='any',axis=0)

#Split df into inputs and target

# Downsample data for speed

columns_to_model = ['PAY_0', 'PAY_2', 'PAY_4', 'LIMIT_BAL', 'PAY_3', 'BILL_AMT1']

df = df.sample(n=1000, random_state=42, axis=0)
 
# Drop NA rows
df = df.dropna(how='any',axis=0)

#Split df into inputs and target
X = df[columns_to_model]
y = df['DEFAULT']

#Create 70/30 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
#initiate and fit Gradient Boosted Classifier
print('Training model...')
log_reg = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
log_reg.fit(X_train, y_train)
 
#Predict test set
print('Evaluating initial model on test data...')
preds = log_reg.predict(X_test)
 
#View performance metrics and save them to domino stats!

m_auc = np.round(roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]), 4)
m_logloss = np.round(log_loss(y_test, log_reg.predict_proba(X_test)[:, 1]), 4)
m_f1_score = np.round(f1_score(y_test, log_reg.predict(X_test)), 4)
m_precision_score = np.round(precision_score(y_test, log_reg.predict(X_test)), 4)
m_recall_score = np.round(recall_score(y_test, log_reg.predict(X_test)), 4)

print("AUC: ", m_auc)
print("logloss: ", m_logloss)
 
#Code to write R2 value and MSE to dominostats value for population in experiment manager
with open('dominostats.json', 'w') as f:
    f.write(json.dumps({"AUC": m_auc,
                       "logloss": m_logloss}))

print('Creating visualizations...')

# Add visualizations and save for inspection
# RocCurveDisplay.from_estimator(log_reg, X_test, y_test, name='Logistic Regression AUC Curve')
roc_curve_display = sklearn.metrics.plot_roc_curve(log_reg, X_test, y_test)
fig = roc_curve_display.figure_
plt.savefig('/mnt/artifacts/log_reg_ROC_Curve.png')

y_pred = log_reg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(cm).plot()

plt.savefig('/mnt/artifacts/log_reg_confusion_matrix.png')

prec, recall, _ = precision_recall_curve(y_test, y_pred,
                                         pos_label=log_reg.classes_[1])

pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

plt.savefig('/mnt/artifacts/log_reg_precision_recall.png')

#Saving trained model to serialized pickle object 
 
import pickle 
 
# save best model
file = '/mnt/code/models/sklearn_logreg.pkl'
pickle.dump(log_reg, open(file, 'wb'))


### Log results in Experiment Tracker ###

experiment = mlflow.set_experiment(experiment_name=os.environ.get('DOMINO_PROJECT_NAME') + " " + os.environ.get('DOMINO_STARTING_USERNAME'))

# Unique name for Experiment Run
now = datetime.now()
dt_string = now.strftime("%m/%d/%Y")

run_name = 'LogReg Tuning ' + dt_string

with mlflow.start_run(run_name=run_name) as parent_run:
    
    print('Logging logistic regression performance to Experiment Tracker')
    
    # Set MLFlow tag to differenciate the model approaches
    mlflow.set_tag("Model_Type", "sklearn")

    # Unique name to log the model with MLFlow
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
 
    model_name = 'Logistic_Regression_Model_{}'.format(dt_string)
    
    # Specify model signature for scoring
    signature = infer_signature(X_test, y_test)
    print(f"model signature: {signature}")
    
    for c in [0.01, 0.1, 1, 10]:
                
        child_run = 'C: {}'.format(str(c))
        
        with mlflow.start_run(run_name=child_run, nested=True) as child_run:
            
            mlflow.set_tag("Model_Type", "sklearn")
            
            log_reg = LogisticRegression(penalty='l1',
                                         solver='liblinear',
                                         C=c,
                                         random_state=42)
            
            log_reg.fit(X_train, y_train)

            #Predict test set
            print('Traiing on C={}'.format(str(c)))
            preds = log_reg.predict(X_test)

            # Score Model
            preds = log_reg.predict(X_test)

            # Calc Model Performance
            m_auc = np.round(roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]), 4)
            m_logloss = np.round(log_loss(y_test, log_reg.predict_proba(X_test)[:, 1]), 4)
            m_f1_score = np.round(f1_score(y_test, log_reg.predict(X_test)), 4)
            m_precision_score = np.round(precision_score(y_test, log_reg.predict(X_test)), 4)
            m_recall_score = np.round(recall_score(y_test, log_reg.predict(X_test)), 4)

            # Model Parameters
            mlflow.log_param("penalty", 'l1')
            mlflow.log_param("solver", 'liblinear')
            mlflow.log_param("C", c)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("model_type", "sklearn")

            # Model Metrics
            mlflow.log_metric("AUC", m_auc)
            mlflow.log_metric("logloss", m_logloss)
            mlflow.log_metric("f1_score", m_f1_score)
            mlflow.log_metric("precision_score", m_precision_score)
            mlflow.log_metric("recall_score", m_recall_score)
            
            mlflow.sklearn.log_model(log_reg,
                         artifact_path="logreg_model",
                         signature=signature)
            
            create_visuals(log_reg, c)
            
    # Model Artifacts
    mlflow.log_artifact('/mnt/artifacts/log_reg_ROC_Curve.png')
    mlflow.log_artifact('/mnt/artifacts/log_reg_confusion_matrix.png')
    mlflow.log_artifact('/mnt/artifacts/log_reg_precision_recall.png')
    mlflow.log_artifact('/mnt/code/models/sklearn_logreg.pkl')

    # Log Model in the Model Registry
    print('Logging logistic regression model to Experiment Tracker')

    mlflow.sklearn.log_model(log_reg,
                             artifact_path="logreg_model",
                             signature=signature)
    
    # You can register your model from here, or from the Domino Experiments Tracker

    # client = mlflow.tracking.MlflowClient()
    # client.create_registered_model(model_name)

mlflow.end_run()
 
print('Script complete!')