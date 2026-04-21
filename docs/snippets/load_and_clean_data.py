def load_and_clean_data(filepath):
    print(f"Reading data from: {filepath}")
    df = pd.read_csv(filepath)

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # empty cells -> NaN -> 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # map main variable to {0,1}
    df['Churn'] = df['Churn']
        .map({'Yes': 1, 'No': 0})

    # one-hot encoding

    df = pd.get_dummies(df, drop_first=True)

    # cast everything needed to float

    df = df.astype(float)

    return df