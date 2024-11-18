###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
 
import streamlit as st
from streamlit import components
import numpy as np
import pandas as pd
import pickle
import time
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns
import requests
import eli5
import xgboost as xgb
import seaborn as sns
import os
 
xgc = xgb.Booster(model_file="/mnt/code/models/xgb_clf.xgb")
 

st.set_page_config(layout="wide")
 
 
    
####################
### INTRODUCTION ###
####################
 
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Credit Card Default Dashboard')
with row0_2:
    st.text("")
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown("")
    
#################
### SELECTION ###
#################
 
 
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
 
### SEASON RANGE ###
st.sidebar.subheader("**Enter the application inputs to view the default scores.**")
st.sidebar.subheader("")
with st.sidebar.form("my_form"):
    PAY_0 = st.number_input('Repayment status last month', min_value = -1, max_value = 9)
    PAY_2 = st.number_input('Repayment status 2 months ago', min_value = -1, max_value = 9)
    PAY_3 = st.number_input('Repayment status 3 months ago', min_value = -1, max_value = 9)
    PAY_4 = st.number_input('Repayment status 4 months ago', min_value = -1, max_value = 9)
    LIMIT_BAL = st.number_input('Credit Limit', min_value=0)
    BILL_AMT1 = st.number_input('Bill Amount', min_value=0)
    scored = st.form_submit_button("Score")
 

column_names = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "LIMIT_BAL", "BILL_AMT1"]
 
# column_names_all = ['PAY_0', 'PAY_2', 'PAY_4', 'LIMIT_BAL', 'PAY_3', 'BILL_AMT1']
    
# Set some default values
sample_data = [[-1, 2, 0, 12000, 0, 2682]]
 
df_all = pd.DataFrame(sample_data, columns=column_names)
 
# Override the values with what was passed to the scoring function
df_all[["PAY_0"]] = PAY_0
df_all[["PAY_2"]] = PAY_2
df_all[["PAY_3"]] = PAY_3
df_all[["PAY_4"]] = PAY_4
df_all[["LIMIT_BAL"]] = LIMIT_BAL
df_all[["BILL_AMT1"]] = BILL_AMT1
 
for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "LIMIT_BAL", "BILL_AMT1"]:
    df_all[col] = df_all[col].astype('int')
 
df = pd.DataFrame(columns=column_names, 
                 data=[[PAY_0, PAY_2, PAY_3, PAY_4, LIMIT_BAL, BILL_AMT1]])
 
setup_dict = {}
scoring_request = {}
results = list()
 
for n in range(df.shape[0]):
    for i in list(df.columns):
        setup_dict.update({i :list(df[n:n+1].to_dict().get(i).values())[0]})
        scoring_request = {'data' : setup_dict}
        
        response = requests.post(os.environ['API_URL'],
    auth=(
        os.environ['API_PASSWORD'],
        os.environ['API_PASSWORD']
    ),
        json=scoring_request
    )
    results.append(response.json().get('result'))
 
 
### Results ###
 
probability = results[0][0]

if probability == 1:
    result_text = ":green[REPAYMENT LIKELY]"
else:
    result_text = ":red[HIGH RISK OF DEFAULT]"
 
try:
    try_df = eli5.explain_prediction_df(xgc, df_all.iloc[0], 
                         feature_names=list(df_all.columns))
except:
    print("An exception occurred")
 
row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
with row4_1:
    st.subheader('After scoring this application, the model suggests that the application be:')
    st.subheader(' ')
    st.subheader(result_text)
    st.subheader(' ')
    st.subheader("The following table provides insights into the model's prediction:")
    st.subheader(' ')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Model Weights:')
        df_weights = eli5.explain_weights_df(xgc, 
                        feature_names=list(df_all.columns))
        df_weights.columns = ['Feature', 'Weight']
        st.dataframe(df_weights.style.background_gradient(axis=0, 
                                                             gmap=df_weights['Weight'], 
                                                             subset=['Feature', 'Weight'], 
                                                             cmap = 'Greens').hide_index())
            
        
    with col2:
        st.subheader('Model Prediction:')
        df_prediction = eli5.explain_prediction_df(xgc, df_all.iloc[0], 
                         feature_names=list(df_all.columns))
        df_prediction.drop("target", axis=1, inplace=True)
        df_prediction.columns = ['Feature', 'Weight', 'value']
        st.dataframe(df_prediction.style.background_gradient(axis=0, 
                                                             gmap=df_prediction['Weight'], 
                                                             subset=['Feature', 'Weight'], 
                                                             cmap = 'RdYlGn').hide_index())
