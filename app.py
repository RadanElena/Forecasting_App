import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

# Python packages
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from streamlit import dataframe, stop
from datetime import datetime


# Import Scikit-Learn
from numpy.random import seed
from sklearn.preprocessing import StandardScaler
import pickle
import sys
from datetime import date
import sqlite3

# Stream lit configuration
st.set_page_config(page_title='Load Forecasting', page_icon=None, layout='wide', initial_sidebar_state='auto')

##################################################################################################################################
                                                
                                                # TSO - APP #
st.sidebar.header('LOAD PREDICTION DATA')
data_collection_list = ['Load database']
data_collection = st.sidebar.multiselect('', data_collection_list)
                                             
st.cache(persist=True)
def data_collection_options():
    if data_collection_list[0] in data_collection:
        # MACHINE-LEARNING DATA
        # dataset\clean_database.db
        sen_file = sqlite3.connect('clean_database.db') 
        df = pd.read_sql("SELECT * FROM Feature_selected_forecasted_data", sen_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df=df.set_index('timestamp')
        
        # SELECT PREDICTION DATE
        col1, col2 = st.columns(2) 
        # with col1:
        start_1 = pd.to_datetime('2021-08-12') 
        end_1 = pd.to_datetime('2022-01-01')

        # Start prediction date - calendar module    
        prediction_start_date = st.sidebar.date_input('Start date',
                                                value=start_1,
                                                min_value = start_1,
                                                max_value = end_1,
                                                key='prediction_date_1')
       
        prediction_start_date = pd.to_datetime(prediction_start_date) 

        # End prediction date - calendar module    
        prediction_end_date = st.sidebar.date_input('End date',
                                            value=start_1 + pd.Timedelta(1, unit='D'),
                                            min_value = start_1,
                                            max_value = end_1,
                                            key='prediction_date_2')

        prediction_end_date = pd.to_datetime(prediction_end_date) 

        # Train-Test split data
        df_train = df[(df.index < prediction_start_date)].copy(deep=True)
        df_test = df[(df.index  >= prediction_start_date) & (df.index < prediction_end_date)].copy(deep=True)

        # Train
        X_train = df_train.drop(columns = ['real_energy_load']).copy()
        y_train = df_train[['real_energy_load']].copy() 

        # Test
        X_test = df_test.drop(columns = ['real_energy_load']).copy()
        y_test = df_test[['real_energy_load']].copy() 

    else:
        st.warning('Please load data')       
        
    return X_train, X_test,  y_train, y_test


# Get Prediction data
if data_collection:
    X_train, X_test,  y_train, y_test = data_collection_options()
    st.subheader("Model input data")  
    st.dataframe(X_test.astype(int))
    
    # Scale data features
    scaler_features = StandardScaler()
    X_train_std = scaler_features.fit_transform(X_train)
    x_test_std = scaler_features.transform(X_test)
    
    # Scale data target
    scaler_target = StandardScaler()
    y_train_std = scaler_target.fit_transform(y_train)
    y_test_std = scaler_target.transform(y_test)



###############################################################################################################################
def model_prediction(df: pd.DataFrame):
    st.subheader("Model results data")  
    
    # Load model
    pickled_model = pickle.load(open('random_forest.pkl', 'rb'))
    y_pred = pickled_model.predict(df)
    y_pred = pd.DataFrame(y_pred, columns =["model_forecast_energy_load"])
    
    # Reverse scaler
    y_pred = scaler_target.inverse_transform(y_pred)
    y_pred = pd.DataFrame(y_pred, columns=['model_forecast_energy_load'])
    y_pred.index = y_test.index
    y_pred['real_energy_load'] = y_test['real_energy_load'].copy()

    return y_pred
##########################################################################################################################

# Run Random Forest model
model_run = st.sidebar.checkbox("Predict Load")

if model_run and data_collection:
    # Make prediction
    y_pred = model_prediction(x_test_std)
    y_pred = y_pred.astype(int)
    y_pred


    # Plot data
    fig = px.line(y_pred)
    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Load Forecast",
        width=1500,
        height=500,
    )

    st.write(fig)
  