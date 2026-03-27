import xgboost as xgb

def get_xgboost_model():
    """
    Initializes and returns an XGBoost model ready for training.
    """
    # We calculate the class weight based on your plot 1:
    # (Number of customers staying / Number leaving) = approx. 5174 / 1869 ≈ 2.76
    # This will help the model handle the minority class (churned customers) better

    model = xgb.XGBClassifier(
        n_estimators=100,        # Number of decision trees
        max_depth=4,             # Maximum tree depth (prevents overfitting)
        learning_rate=0.05,      # Learning rate
        subsample=0.8,           # Uses 80% of data to build each tree
        colsample_bytree=0.8,    # Uses 80% of features to build each tree
        scale_pos_weight=2.76,   # Class balancing!
        eval_metric='logloss',   # Evaluation metric during training
        random_state=42,         # Ensures reproducibility
        n_jobs=-1                # Uses all CPU cores
    )

    return model
