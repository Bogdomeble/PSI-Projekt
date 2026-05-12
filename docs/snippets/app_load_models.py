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