
 

import streamlit as st 

import pandas as pd 

import joblib 

 

model = joblib.load('churn_model.pkl') 

model_columns = joblib.load('model_columns.pkl') 

threshold = joblib.load('threshold.pkl') 
st.title("Customer Churn Prediction App") 

 

 

st.write('Enter customer details:') 

 

total_day_charge = st.number_input( 

    "Total Day Charge", 

    min_value=0.0,

    value=40.0

) 

 

total_day_min = st.number_input( 

    "Total Day Minutes", 

    min_value=0.0,

    value=230.0

) 

 

customer_service_calls = st.number_input( 

    "Customer Service Calls", 

    min_value=0, 

    value=4

) 

 

int_plan = st.selectbox( 

    "International Plan", 

    ['Yes','No'] 

) 

 

if int_plan == 'Yes': 

  int_plan = 1 

else: 

  int_plan = 0 

 

 

total_eve_min = st.number_input( 

    "Total Evening Minutes", 

    min_value=0,

    value=210 

) 

 

total_eve_chg = st.number_input( 

    "Total Evening Charge", 

    min_value=0, 

    value=20 

) 

 

 

input_data = pd.DataFrame( 

    [[ 

        total_day_charge, 

        total_day_min, 

        customer_service_calls, 

        int_plan, 

        total_eve_min, 

        total_eve_chg 

         

    ]], columns = model_columns 

) 

 

st.write("Input given data to model:") 

st.dataframe(input_data) 

 

if st.button("Predict Churn"): 

  churn_pb = model.predict(input_data)[0][1] 

 

  if churn_pb >= threshold: 

    prediction = 1 

  else: 

    prediction = 0 

 

 

  st.write("Churn Probability:", round(churn_pb * 100,2), "%") 

  st.write("Threshold:", threshold) 

 

  if prediction == 1: 

    st.error("Final Prediction : customer may churn") 

  else:  

    st.success("Customer may not churn") 

 
