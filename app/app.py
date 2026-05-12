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

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.markdown(
    """
<style>
    header.stAppHeader { background-color: transparent; }
    section.stMain .block-container { padding-top: 1rem; }
    .stAppDeployButton { display:none }
    .dataframe { font-size: 14px; }
</style>""",
    unsafe_allow_html=True,
)

if 'nn_prob' not in st.session_state:
    st.session_state.nn_prob = None
    st.session_state.xgb_prob = None
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


def calc_churn():
    # create empty dataframe with the same shape and size
    input_df = pd.DataFrame(0, index=[0], columns=full_columns)

    # input data values from streamlit form
    input_df['tenure'] = st.session_state['tenure']
    input_df['MonthlyCharges'] = st.session_state['monthly_charges']
    input_df['TotalCharges'] = st.session_state['total_charges']
    input_df['SeniorCitizen'] = 1.0 if "Senior" in st.session_state['miscdata'] else 0.0

    # manual one-hot encoding - this needs to be changed for scaling purposes
    # for every choice/column append data to dataframe
    # if said column exists - set it to 1

    categorical_choices = {
        "Contract": st.session_state['contract'],
        "InternetService": st.session_state['internet'],
        "TechSupport": st.session_state['tech_support'],
        "PaymentMethod":  st.session_state['payment_method'],
        "gender" : st.session_state['gender'],
        "OnlineSecurity": st.session_state['online_security'],
        "OnlineBackup": st.session_state['online_backup'],
        "DeviceProtection": st.session_state['device_protection'],
        "StreamingTV": st.session_state['stream_tv'],
        "StreamingMovies": st.session_state['stream_movie'],
        "MultipleLines": st.session_state['multiple_lines'],
        "Partner" : "Yes" if "Has a Partner" in st.session_state['miscdata'] else "No",
        "Dependents" : "Yes" if "Has Dependents" in st.session_state['miscdata'] else "No",
        "PhoneService": "Yes" if st.session_state['phone_service'] else "No",
        "PaperlessBilling": "Yes" if st.session_state['paperless_billing'] else "No",
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

    st.session_state.nn_prob = nn_prob
    st.session_state.xgb_prob = xgb_prob



# simple user form
st.title("Panel Predykcji Odejścia Klienta (Churn)")
st.write("Wprowadź dane klienta, aby sprawdzić, czy zrezygnuje z usług naszej sieci.")

config_side, output_side = st.columns([3, 1])
with config_side:
    col1, col2, col3 = st.columns(3)


    with col1:
        tenure = st.slider(
            "Czas trwania umowy (miesiące)", 0, 72, 12,
            key='tenure'
        )

    with col2:
        internet = st.selectbox(
            "Internet",["DSL", "Fiber optic", "No"],
            key='internet'
        )
    with col3:
        contract = st.selectbox(
            "Typ kontraktu",["Month-to-month", "One year", "Two year"],
            key='contract'
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        monthly_charges = st.number_input(
            "Miesięczne opłaty ($)", min_value=0.0, max_value=200.0, value=50.0,step=1.0,
            key='monthly_charges'

        )
    with col2:
        total_charges = st.number_input(
            "Całkowite opłaty ($)", min_value=0.0, max_value=10000.0, value=500.0,step=1.0,
            key='total_charges'
        )
    with col3:
        payment_method = st.selectbox("Metoda płatności",[
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], key='payment_method'
        )

    col1, col2, _ = st.columns(3)
    with col1:
        client_info_pills = st.pills(
            "Dane o Kliencie",
            ["Senior", "Has a Partner", "Has Dependents"],
            selection_mode="multi", key='miscdata'
        )
    with col2:
        gender = st.segmented_control(
            "Płeć", ["Male", "Female"], default="Male",
            key='gender'
        )

    col1, col2, col3,col4 = st.columns(4)
    y_n_nis = ["Yes", "No", "No internet service"]
    with col1:
        tech_support = st.segmented_control(
            "Wsparcie techniczne?",y_n_nis,
            key='tech_support', default="No"
        )

    with col2:
        online_sec = st.segmented_control(
            "Ochrona Online?",y_n_nis,
            key='online_security', default="No"
        )

    with col3:
        online_backup = st.segmented_control(
            "Online Backup?",y_n_nis,
            key='online_backup', default="No"
        )

    with col4:
        device_protection = st.segmented_control(
            "Ochrona urządzeń?",y_n_nis,
            key='device_protection', default="No"
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stream_tv = st.segmented_control(
            "Streaming TV?",y_n_nis,
            key='stream_tv', default="No"
        )

    with col2:
        stream_movie = st.segmented_control(
            "Streaming filmów?",y_n_nis,
            key='stream_movie', default="No"
        )
    with col3:
        multiple_lines = st.segmented_control(
            "Kilka linii?", ["Yes", "No", "No phone service"],
            key='multiple_lines', default="No"
        )
    with col4:
        phoneService = st.checkbox(
            "Phone service?", key="phone_service"
        )
        paperlessBilling = st.checkbox(
            "Paperless billing?", key="paperless_billing"
        )


    # process data with st.button
    st.button("Przewiduj Churn", type="primary", use_container_width=True, on_click=calc_churn)

    with output_side:
        nn_prob = st.session_state['nn_prob']
        if(nn_prob is not None):

            a, b = st.columns(2)
            # PyTorch
            st.write("**Sieć Neuronowa (PyTorch)**")
            st.metric(label="Ryzyko odejścia", value=f"{nn_prob*100:.1f}%")
            if nn_prob > 0.4: # the modified threshold
                st.error("Wysokie ryzyko odejścia!")
            else:
                st.success("Klient raczej zostanie.")
            xgb_prob = st.session_state['xgb_prob']
            # XGBoost
            st.write("**XGBoost Classifier**")
            st.metric(label="Ryzyko odejścia", value=f"{xgb_prob*100:.1f}%")
            if xgb_prob > 0.5:
                st.error("Wysokie ryzyko odejścia!")
            else:
                st.success("Klient raczej zostanie.")
