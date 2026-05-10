# app.py

import sys
import os

#this fixes path errors when using streamlit run app/app.py - maybe we need to change the name here?

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import torch
import joblib
import xgboost as xgb
import json
import numpy as np

# pytorch model import
from src.models.neural_net import ChurnNeuralNet

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

# --- load models and scaler - only once using @st.cache_resource ---
@st.cache_resource
def load_assets():
    # load tensors for both models
    with open("exported-models/columns.json", "r") as f:
        full_columns = json.load(f)

   
    with open("exported-models/xgb_columns.json", "r") as f:
        xgb_columns = json.load(f)
        
    scaler = joblib.load("exported-models/scaler.pkl")
    
    # load trained pytorch model (check dimension)
    input_dim = len(full_columns)
    nn_model = ChurnNeuralNet(input_dim)
    nn_model.load_state_dict(torch.load("exported-models/nn_model.pt", map_location=torch.device('cpu')))
    nn_model.eval()
    
    # load XGBoost classifier from xgb_model.json
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("exported-models/xgb_model.json")
    
    return nn_model, xgb_model, scaler, full_columns, xgb_columns

nn_model, xgb_model, scaler, full_columns, xgb_columns = load_assets()

# simple user form
st.title("Panel Predykcji Odejścia Klienta (Churn)")
st.write("Wprowadź dane klienta, aby sprawdzić, czy zrezygnuje z usług naszej sieci.")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Czas trwania umowy (miesiące)", 0, 72, 12)
    monthly_charges = st.number_input("Miesięczne opłaty ($)", min_value=0.0, max_value=200.0, value=50.0,step=1.0)
    total_charges = st.number_input("Całkowite opłaty ($)", min_value=0.0, max_value=10000.0, value=500.0,step=1.0)
    contract = st.selectbox("Typ kontraktu",["Month-to-month", "One year", "Two year"])

with col2:
    internet = st.selectbox("Internet",["DSL", "Fiber optic", "No"])
    tech_support = st.radio("Wsparcie techniczne?",["Yes", "No", "No internet service"])
    payment_method = st.selectbox("Metoda płatności",[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

# process data with st.button
if st.button("🔮 Przewiduj Churn", type="primary", use_container_width=True):
    
    # create empty dataframe with the same shape and size
    input_df = pd.DataFrame(0, index=[0], columns=full_columns)
    
    # input data values from streamlit form
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges
    input_df['TotalCharges'] = total_charges
    
    # manual one-hot encoding - this needs to be changed for scaling purposes
    # for every choice/column append data to dataframe 
    # if said column exists - set it to 1

    categorical_choices = {
        "Contract": contract,
        "InternetService": internet,
        "TechSupport": tech_support,
        "PaymentMethod": payment_method
    }
    
    for feature, value in categorical_choices.items():
        dummy_name = f"{feature}_{value}"

        # this fixes the drop_first=true problem - just create an empty array :)))))

        if dummy_name in input_df.columns:
            input_df[dummy_name] = 1
            
    # check if column size is correct
    input_df = input_df[full_columns] 

    # scale the data using standard scaler
    input_scaled = scaler.transform(input_df.values)

    # --- predykcja ---
    
    # pytorch uses all 30 variables
    X_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        nn_output = nn_model(X_tensor)
        nn_prob = torch.sigmoid(nn_output).item()

    # xgboost only needs the 18 variables here
    xgb_indices =[full_columns.index(col) for col in xgb_columns]
    xgb_input_scaled = input_scaled[:, xgb_indices] 
    
    xgb_prob = xgb_model.predict_proba(xgb_input_scaled)[0][1]

    # --- results ---
    st.markdown("---")
    st.subheader("Wyniki modeli:")
    
    res_col1, res_col2 = st.columns(2)
    
    # PyTorch
    with res_col1:
        st.write("**Sieć Neuronowa (PyTorch)**")
        st.metric(label="Ryzyko odejścia", value=f"{nn_prob*100:.1f}%")
        if nn_prob > 0.4: # the modified threshold
            st.error("Wysokie ryzyko odejścia!")
        else:
            st.success("Klient raczej zostanie.")

    # XGBoost
    with res_col2:
        st.write("**XGBoost Classifier**")
        st.metric(label="Ryzyko odejścia", value=f"{xgb_prob*100:.1f}%")
        if xgb_prob > 0.5:
            st.error("Wysokie ryzyko odejścia!")
        else:
            st.success("Klient raczej zostanie.")
    
    # build raw dataframe from input data
    input_dict = {
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges":[total_charges],
        "Contract": [contract],
        "InternetService": [internet],
        "TechSupport": [tech_support],
        "PaymentMethod": [payment_method]
    }
    input_df = pd.DataFrame(input_dict)
    
    # One-Hot Encoding
    input_df = pd.get_dummies(input_df)
    
    # fill empty columns with null-values
    for col in full_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # check sizes before and after training
    input_df = input_df[full_columns] 

    # scale the data with the standardScaler (Skaler wymaga 30 cech!)
    input_scaled = scaler.transform(input_df)

    # --- inference ---
    
    # pytorch uses all 30 variables
    X_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        nn_output = nn_model(X_tensor)
        nn_prob = torch.sigmoid(nn_output).item()

    # xgboost only need the slim 18 variables
    # get the indexes from the columns
    xgb_indices = [full_columns.index(col) for col in xgb_columns]
    xgb_input_scaled = input_scaled[:, xgb_indices] # get the needed 18 columns
    
    xgb_prob = xgb_model.predict_proba(xgb_input_scaled)[0][1]


    # results
    st.markdown("---")
    st.subheader("Wyniki modeli:")
    
    res_col1, res_col2 = st.columns(2)
    
    # print pytorch results
    with res_col1:
        st.write("**Sieć Neuronowa (PyTorch)**")
        st.metric(label="Ryzyko odejścia", value=f"{nn_prob*100:.1f}%")
        if nn_prob > 0.4: # threshold from the tweaks
            st.error("Wysokie ryzyko odejścia!")
        else:
            st.success("Klient raczej zostanie.")

    #  XGboost results
    with res_col2:
        st.write("**XGBoost Classifier**")
        st.metric(label="Ryzyko odejścia", value=f"{xgb_prob*100:.1f}%")
        if xgb_prob > 0.5:
            st.error("Wysokie ryzyko odejścia!")
        else:
            st.success("Klient raczej zostanie.")


# 
# todo: fix UserWarning: X has feature names, but StandardScaler was fitted without feature names - warnings.warn( ??????

# 