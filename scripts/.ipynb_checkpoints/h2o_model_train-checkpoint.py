import h2o
from h2o.automl import H2OAutoML
import json
import pickle 
import pandas as pd
import random
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os
import mlflow
import mlflow.h2o

# Set train-test split to 70
n = 70

# Read in the new dataset
path = str('/mnt/data/mlops-best-practices/credit_card_default.csv')
data = pd.read_csv(path)
print('Read in {} rows of data'.format(data.shape[0]))

# Find all Pearson correlations of numerical variables with the target variable, assuming it's 'DEFAULT'
corr_values = data.corr(numeric_only=True).sort_values(by='DEFAULT')['DEFAULT'].drop('DEFAULT', axis=0)

# Keep all variables with above an 8% Pearson correlation
important_feats = corr_values[abs(corr_values) > 0.08]

# Prepare the dataset for model training and evaluation

# Drop NA rows
data = data.dropna(how='any', axis=0)

# Select important features and the target variable
data = data[list(important_feats.keys()) + ['DEFAULT']]

# Split the data into training and testing sets
train = data[0:round(len(data) * n / 100)]
test = data[train.shape[0]:]

print('H2O version -{}'.format(h2o.__version__))

# Initialize local H2O
h2o.init()

# Set up a new MLFlow experiment
mlflow.set_experiment(experiment_name=os.environ.get('DOMINO_PROJECT_NAME') + " " + os.environ.get('DOMINO_STARTING_USERNAME') + " " + os.environ.get('MLFLOW_NAME'))

# Convert data to H2O frames
hTrain = h2o.H2OFrame(train)
hTest = h2o.H2OFrame(test)

# Identify predictors and response
x = hTrain.columns
y = "DEFAULT"
x.remove(y)

# Train the model and log metrics with MLFlow
with mlflow.start_run():
    mlflow.set_tag("Model_Type", "H2O AutoML")

    # Run AutoML for 5 base models (limited to 1 min max runtime)
    print('Training AutoML model...')
    aml = H2OAutoML(max_models=10, max_runtime_secs=30, sort_metric="r2")
    aml.train(x=x, y=y, training_frame=hTrain)

    # Evaluate the model on the validation data
    best_model = aml.leader
    preds = best_model.predict(hTest)

    # Try to get R² and MSE, handle NoneType if not available
    r2 = best_model.r2(xval=True)
    mse = best_model.mse(xval=True)

    if r2 is not None:
        r2 = round(r2, 3)
    else:
        print("R² metric not available for this model.")
        r2 = "N/A"

    if mse is not None:
        mse = round(mse, 3)
    else:
        print("MSE metric not available for this model.")
        mse = "N/A"

    print("R2 Score:", r2)
    print("MSE:", mse)

    # Log metrics in MLFlow if available
    if isinstance(r2, (int, float)):
        mlflow.log_metric("R2", r2)
    if isinstance(mse, (int, float)):
        mlflow.log_metric("MSE", mse)

    # Save the metrics for Domino stats
    with open('dominostats.json', 'w') as f:
        f.write(json.dumps({"R2": r2, "MSE": mse}))

    # Write results to a dataframe for visualization
    results = pd.DataFrame({'Actuals': test.DEFAULT.reset_index()['DEFAULT'], 'Predictions': preds.as_data_frame()['predict']})

    print('Creating visualizations...')
    # Scatter plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plt.title('H2O Actuals vs Predictions Scatter Plot')
    sns.regplot(data=results, x='Actuals', y='Predictions', order=3)
    plt.savefig('/mnt/artifacts/actual_v_pred_scatter.png')
    mlflow.log_figure(fig1, 'actual_v_pred_scatter.png')

    # Histogram
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plt.title('H2O Actuals vs Predictions Histogram')
    plt.xlabel('Default Payment')
    sns.histplot(results, bins=6, multiple='dodge', palette='coolwarm')
    plt.savefig('/mnt/artifacts/actual_v_pred_hist.png')
    mlflow.log_figure(fig2, 'actual_v_pred_hist.png')

    # Save the trained model
    h2o.save_model(best_model, path='/mnt/code/models')

mlflow.end_run()
print('Script complete!')
