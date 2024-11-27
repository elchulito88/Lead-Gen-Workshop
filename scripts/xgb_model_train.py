import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import roc_auc_score, log_loss, f1_score, precision_score, recall_score, precision_recall_curve, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
import json
import os
import xgboost as xgb
from datetime import datetime
import pickle

# Import mlflow
import mlflow
import mlflow.xgboost
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.models.signature import infer_signature

def create_visuals(model, param):
    
    print('Creating visualizations for param {}'.format(str(param)))

    # Add visualizations and save for inspection

    # RocCurveDisplay.from_estimator(model, X_test, y_test, name='XGBoost AUC Curve')
    roc_curve_display = sklearn.metrics.plot_roc_curve(model, X_test, y_test)
    fig = roc_curve_display.figure_
    plt.savefig('/mnt/artifacts/xgb_ROC_Curve_max_depth={}.png'), str(param)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    cm_display = ConfusionMatrixDisplay(cm).plot()

    plt.savefig('/mnt/artifacts/xgb_confusion_matrix_max_depth={}.png'), str(param)

    prec, recall, _ = precision_recall_curve(y_test, y_pred,
                                             pos_label=model.classes_[1])

    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

    plt.savefig('/mnt/artifacts/xgb_precision_recall_max_depth={}.png'), str(param)
    
    mlflow.log_artifact('/mnt/artifacts/xgb_ROC_Curve_max_depth={}.png'), str(param)
    mlflow.log_artifact('/mnt/artifacts/xgb_confusion_matrix_max_depth={}.png'), str(param)
    mlflow.log_artifact('/mnt/artifacts/xgb_precision_recall_max_depth={}.png'), str(param)    
    
    # Read in data then split into train and test
path = str('/mnt/data/mlops-best-practices/credit_card_default.csv')
df = pd.read_csv(path)
print('Read in {} rows of data'.format(df.shape[0]))
  
#Get data set up for model training and evaluation
 
# Drop NA rows
df = df.dropna(how='any',axis=0)

# Downsample data for speed

columns_to_model = ['PAY_0', 'PAY_2', 'PAY_4', 'LIMIT_BAL', 'PAY_3', 'BILL_AMT1']

df = df.sample(n=1000, random_state=42, axis=0)
 
# Drop NA rows
df = df.dropna(how='any',axis=0)

# Split df into inputs and target
X = df[columns_to_model]
y = df['DEFAULT']

# Create 70/30 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Initiate and fit initial XGBoost Model
print('Training model...')

# Shuffle parameters for Experiment Tracker
learning_rate = np.random.choice([0.01, 0.05, 0.1])
n_estimators = np.random.choice([100, 150, 200])

xgb_clf = xgb.XGBClassifier(learning_rate = learning_rate,
                            max_depth = 3,
                            n_estimators = n_estimators)

xgb_clf.fit(X_train, y_train)
 
# Predict test set
print('Evaluating model on test data...')
preds = xgb_clf.predict(X_test)
 
# View initial model performance metrics and save them to domino stats!

m_auc = np.round(roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1]), 4)
m_logloss = np.round(log_loss(y_test, xgb_clf.predict_proba(X_test)[:, 1]), 4)
m_f1_score = np.round(f1_score(y_test, xgb_clf.predict(X_test)), 4)
m_precision_score = np.round(precision_score(y_test, xgb_clf.predict(X_test)), 4)
m_recall_score = np.round(recall_score(y_test, xgb_clf.predict(X_test)), 4)

print("AUC: ", m_auc)
print("logloss: ", m_logloss)
 
#Code to write R2 value and MSE to dominostats value for population in experiment manager
with open('dominostats.json', 'w') as f:
    f.write(json.dumps({"AUC": m_auc,
                       "logloss": m_logloss}))
 
print('Creating visualizations...')

# Add visualizations and save for inspection

# RocCurveDisplay.from_estimator(xgb_clf, X_test, y_test, name='XGBoost AUC Curve')
roc_curve_display = sklearn.metrics.plot_roc_curve(xgb_clf, X_test, y_test)
fig = roc_curve_display.figure_
plt.savefig('/mnt/artifacts/xgb_ROC_Curve.png')

y_pred = xgb_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(cm).plot()

plt.savefig('/mnt/artifacts/xgb_confusion_matrix.png')

prec, recall, _ = precision_recall_curve(y_test, y_pred,
                                         pos_label=xgb_clf.classes_[1])

pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

plt.savefig('/mnt/artifacts/xgb_precision_recall.png')

# Same trained model to Project File System

file = '/mnt/code/models/xgb_clf.pkl'
pickle.dump(xgb_clf, open(file, 'wb'))

### Log results in Experiment Tracker ###

experiment = mlflow.set_experiment(experiment_name=os.environ.get('DOMINO_PROJECT_NAME') + " " + os.environ.get('DOMINO_STARTING_USERNAME'))

# Unique name for Experiment Run
now = datetime.now()
dt_string = now.strftime("%m/%d/%Y")

run_name = 'XGBoost Tuning ' + dt_string

with mlflow.start_run(run_name=run_name) as parent_run:
    
    print('Logging XGBoost performance to Experiment Tracker')
    
    # Set MLFlow tag to differenciate the model approaches
    mlflow.set_tag("Model_Type", "xgb")
 
    model_name = 'XGBoost_Model_{}'.format(dt_string)
    
    # Specify model signature for scoring
    signature = infer_signature(X_test, y_test)
    print(f"model signature: {signature}")
    
    for max_depth in [3, 4, 5, 6]:
                
        child_run = 'max_depth: {}'.format(str(max_depth))
        
        with mlflow.start_run(run_name=child_run, nested=True) as child_run:
            
            mlflow.set_tag("Model_Type", "xgb")
        
            xgb_clf = xgb.XGBClassifier(learning_rate = learning_rate,
                                        max_depth = max_depth,
                                        n_estimators = n_estimators)

            xgb_clf.fit(X_train, y_train)

            # Score Model
            print('Traiing on max depth of {}'.format(str(max_depth)))
            preds = xgb_clf.predict(X_test)

            # Calc Model Performance
            m_auc = np.round(roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1]), 4)
            m_logloss = np.round(log_loss(y_test, xgb_clf.predict_proba(X_test)[:, 1]), 4)
            m_f1_score = np.round(f1_score(y_test, xgb_clf.predict(X_test)), 4)
            m_precision_score = np.round(precision_score(y_test, xgb_clf.predict(X_test)), 4)
            m_recall_score = np.round(recall_score(y_test, xgb_clf.predict(X_test)), 4)

            # Model Parameters
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("model_type", "XGBoost")

            # Model Metrics
            mlflow.log_metric("AUC", m_auc)
            mlflow.log_metric("logloss", m_logloss)
            mlflow.log_metric("f1_score", m_f1_score)
            mlflow.log_metric("precision_score", m_precision_score)
            mlflow.log_metric("recall_score", m_recall_score)
            
            mlflow.xgboost.log_model(xgb_clf,
                         artifact_path="xgb_model",
                         signature=signature,
                         )
            
            create_visuals(xgb_clf, max_depth)
            
    # Model Artifacts
    mlflow.log_artifact('/mnt/artifacts/xgb_ROC_Curve.png')
    mlflow.log_artifact('/mnt/artifacts/xgb_confusion_matrix.png')
    mlflow.log_artifact('/mnt/artifacts/xgb_precision_recall.png')
    mlflow.log_artifact('/mnt/code/models/xgb_clf.pkl')

    # Log Model in the Model Registry
    print('Logging XGBoost model to Experiment Tracker')

    mlflow.xgboost.log_model(xgb_clf,
                             artifact_path="xgb_model",
                             signature=signature,
                             )
    
    # You can register your model from here, or from the Domino Experiments Tracker
    
    # client = mlflow.tracking.MlflowClient()
    # client.create_registered_model(model_name)

mlflow.end_run()
 
print('Script complete!')