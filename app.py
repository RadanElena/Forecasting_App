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
from numpy.random import seed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import plotly.figure_factory as ff
import pickle
import sys
from datetime import date
import sqlite3

# Stream lit configuration
st.set_page_config(page_title='Load Forecasting', page_icon=None, layout='wide', initial_sidebar_state='auto')

##################################################################################################################################
                                                
                                                # TSO - APP #
st.sidebar.header('LOAD PREDICTION DATA')
# data_collection_list = ['Load database']
# data_collection = st.sidebar.multiselect('', data_collection_list)
                                             
st.cache(persist=True)

def data_collection_options():

    # if data_collection_list[0] in data_collection:
        # MACHINE-LEARNING DATA
        # dataset\clean_database.db

    conn = sqlite3.connect('clean_database.db') 
    df = pd.read_sql("SELECT * FROM Feature_selected_forecasted_data", conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
  
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
                                        value = prediction_start_date + pd.Timedelta(1, unit='D'),
                                        min_value = prediction_start_date + pd.Timedelta(1, unit='D'),
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


    # else:
    #     st.warning('Please load data')       
        
    return X_train, X_test,  y_train, y_test 


# Get Prediction data
X_train, X_test,  y_train, y_test = data_collection_options()

st.subheader("Model input data")  
st.dataframe(X_test.astype(int))

# Scale data features
scaler_features = StandardScaler()
X_train_std = scaler_features.fit_transform(X_train)
X_test_std = scaler_features.transform(X_test)

# Scale data target
scaler_target = StandardScaler()
y_train_std = scaler_target.fit_transform(y_train)
y_test_std = scaler_target.transform(y_test)

###############################################################################################################################

# Models prediction 
def model_prediction(df: pd.DataFrame):
    st.subheader("Model results data")  
    
    # Load model
    pickled_model = pickle.load(open('random_forest.pkl', 'rb'))
    y_pred = pickled_model.predict(df)
    y_pred = pd.DataFrame(y_pred, columns =["Model_forecast [MW]"])
    
    # Reverse scaler
    y_pred = scaler_target.inverse_transform(y_pred)
    y_pred = pd.DataFrame(y_pred, columns=['Model_forecast [MW]'])
    y_pred.index = y_test.index

    conn = sqlite3.connect('clean_database.db')

    enstoe_forecast = pd.read_sql("SELECT * FROM Forecasted_energy_load_by_Entsoe", conn)
    enstoe_forecast = enstoe_forecast.rename(columns={"forecasted_energy_load": "Entsoe_forecast [MW]"})
    enstoe_forecast['timestamp'] = pd.to_datetime(enstoe_forecast['timestamp'])
    enstoe_forecast = enstoe_forecast.set_index('timestamp')
    y_pred = pd.concat([y_pred, enstoe_forecast], axis = 1)
    y_pred = y_pred.dropna()

    y_pred['Real_energy_load [MW]'] = y_test['real_energy_load'].copy()

    return y_pred

##########################################################################################################################

# Models metrics
def model_metrics(df: pd.DataFrame):
    st.subheader("Regression metrics") 

    y_predict = df.iloc[:,:1]
    y_entsoe = df.iloc[:, 1:2]
    y_true = df.iloc[:, 2:3]

    model_mae = mean_absolute_error(y_true = y_true, y_pred = y_predict)
    entsoe_mae = mean_absolute_error(y_true = y_true, y_pred = y_entsoe)

    model_rmse = mean_squared_error(y_true = y_true, y_pred = y_predict, squared=False)
    entsoe_rmse = mean_squared_error(y_true = y_true, y_pred = y_entsoe, squared=False)

    model_mape = mean_absolute_percentage_error(y_true = y_true, y_pred = y_predict)
    entsoe_mape = mean_absolute_percentage_error(y_true = y_true, y_pred = y_entsoe)

    model_max_error = max_error(y_true = y_true, y_pred = y_predict)
    entsoe_max_error = max_error(y_true = y_true, y_pred = y_entsoe)

    model_r2 = r2_score(y_true = y_true, y_pred = y_predict)
    entsoe_r2 = r2_score(y_true = y_true, y_pred = y_entsoe)

    d = {'Model_metrics': [model_mae, model_rmse, model_mape, model_max_error, model_r2], 'Entsoe_metrics': [entsoe_mae, entsoe_rmse, entsoe_mape, entsoe_max_error, entsoe_r2]}

    metrics_results = pd.DataFrame(data=d, index=["MAE", "RMSE", "MAPE", "MAX ERROR", "R2"])

    return metrics_results

##########################################################################################################################

def data_visualization(df: pd.DataFrame):
    st.subheader("Data Visualization")
    
    # Line-plot with each column
    fig1 = px.line(df)
    fig1.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Load Forecast",
        width=1100,
        height=500,
    )
    st.write(fig1)

    # Box-plot with each column
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y= df["Model_forecast [MW]"], name='Model',
                    marker_color = 'indianred'))

    fig_box.add_trace(go.Box(y= df["Entsoe_forecast [MW]"], name = 'Entsoe',
                    marker_color = 'lightseagreen'))

    fig_box.add_trace(go.Box(y= df["Real_energy_load [MW]"], name = 'Real',
                    marker_color = 'hotpink'))

    fig_box.update_layout(
        width=1025,
        height=500,
    )
    st.write(fig_box)

    # Feature importance
    regr = pickle.load(open('random_forest.pkl', 'rb'))

    feature_importance = regr.feature_importances_
    sorted_idx = np.argsort(feature_importance)

    fig_importance = px.bar(sorted_idx, x=regr.feature_importances_[sorted_idx], y= X_train.columns[sorted_idx])
    fig_importance.update_layout(
        xaxis_title="Feature importance",
        yaxis_title = "Features",
        width=1100,
        height=500,
        )
    st.write(fig_importance)

    # Feature permutation
    perm_importance = permutation_importance(regr, X_test_std, y_test_std)
    sorted_idx = perm_importance.importances_mean.argsort()
    
    fig_permutation = px.bar(sorted_idx, x = perm_importance.importances_mean[sorted_idx], y= X_test.columns[sorted_idx])
   
    fig_permutation.update_layout(
        xaxis_title="Feature Permutation",
        yaxis_title = "Features",
        width=1100,
        height=500,
        )
    st.write(fig_permutation)


##################################################################################################################################
# Run Random Forest model
model_run = st.sidebar.checkbox("Predict Load")

if model_run:
    # Make prediction
    y_pred = model_prediction(X_test_std)
    y_pred = y_pred.astype(int)
    y_pred

    # Calculate the errors
    metrics = model_metrics(y_pred)
    metrics

    data_visualization(y_pred)

  