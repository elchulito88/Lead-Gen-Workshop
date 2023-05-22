import pandas as pd
import numpy as np
import random
import math
import pickle
import json
import os
import requests
import datetime
import boto3
from botocore.exceptions import NoCredentialsError
from domino.training_sets import TrainingSetClient, model

#set DMM vars
bucket = 'wine-quality-monitoring'

#Load in data

print('Reading in data for batch scoring')
df = TrainingSetClient.get_training_set_version('winequality-training-domino-admin', number = 4).load_raw_pandas()

df2 = df.append(df).reset_index(drop=True)

##For each input feature adjust data and round/cast as necessary
#Density - 50%-150
densityJitter = df2.density.apply(lambda x : x*(random.randrange(50,150))/100).round(4)
#volatile acidity - 70%-130%
volatileAcidityJitter = df2.volatile_acidity.apply(lambda x : x*(random.randrange(70,130)/100)).round(2)
#Chlorides - 80%-120%
chloridesJitter = df2.chlorides.apply(lambda x : x*(random.randrange(80,120)/100)).round(3)
#is_red - 40%-160%
is_redJitter = df2.is_red.apply(lambda x : x*(random.randrange(40,160)/100)).round(0)
#alcohol - 90%-110%
alcoholJitter = df2.alcohol.apply(lambda x : x*(random.randrange(90,110)/100)).round(1)

#Take all the new 'jittered' variables and write to a new df
#Keep original custid and churn_Y fields
df3 = pd.DataFrame({'id': df2.id,
       'density': densityJitter, 
       'volatile_acidity': volatileAcidityJitter,
       'chlorides': chloridesJitter,
       'is_red': is_redJitter,
       'alcohol': alcoholJitter,
       'quality': df2.quality
                   })

#Grab between 50 and 500 random rows from jittered data
df_inf = df3.sample(n = random.randint(50,100)).reset_index(drop=True)

#set up clean customer_ids
setup_ids = list(range(0, df_inf.shape[0]))
ids = list()
for i in setup_ids:
    ids.append(str(datetime.date.today())+'_'+str(setup_ids[i]))
    
df_inf['wine_id']=ids    
print('Sending {} records to model API endpoint for scoring'.format(df_inf.shape[0]))

#Set up dictionaries and lists for loops
setup_dict = {}
scoring_request = {}
results = list()

inputs = df_inf[['wine_id','density', 'volatile_acidity', 'chlorides', 'is_red', 'alcohol']]

for n in range(inputs.shape[0]):
    for i in list(inputs.columns):
        setup_dict.update({i :list(inputs[n:n+1].to_dict().get(i).values())[0]})
        scoring_request = {'data' : setup_dict}
        
        
    response = requests.post("https://rev4mlops.domino-eval.com:443/models/646b7aa803e97e4d244dba2a/latest/model",
    auth=(
        "3OOzvWW9oCneKZnagRZ6ac4lbZ3mseyIxeOB3Yk2XvznYly3tY89UKQgDoatkm2H",
        "3OOzvWW9oCneKZnagRZ6ac4lbZ3mseyIxeOB3Yk2XvznYly3tY89UKQgDoatkm2H"
    ),
        json=scoring_request
    )
    results.append(response.json().get('result').get('prediction'))
    
print('Scoring complete')

df_ground_truth=df_inf[['wine_id', 'quality']].rename({'wine_id': 'event_id', 'quality' : 'quality_GT'}, axis=1)
print(df_ground_truth.shape[0]==inputs.shape[0])
print((df_ground_truth.event_id==inputs.wine_id).sum()==df_ground_truth.shape[0])

gt_file_name = str('GT_Data_') + str(datetime.date.today())+str('.csv')
gt_file_path = str('/domino/datasets/local/ground_truth_data/')+gt_file_name
df_ground_truth.to_csv(gt_file_path, index=False)

def s3_upload(local_file, bucket):
    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                      aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    
    s3_file_name = '{}'.format(os.path.basename(local_file))
    
    try:
        s3.upload_file(local_file, bucket, s3_file_name)
        print(str(s3_file_name) + " Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    
s3_upload(gt_file_path, bucket)

print('Data Uploaded to s3 bucket at s3://{}/{}'.format(bucket, gt_file_name))
print('Done!')
