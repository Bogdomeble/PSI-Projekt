# docs/snippets/xgboost_feature_selection.py
def train_and_eval_xgboost_with_feature_selection(
    xgb_model, xgb_data, drop_percent=30, verbose=False
):
    print("\n--- Starting XGBoost Training with feature selection ---")

    # First training to determine feature importance

    features = xgb_data["feature_names"]
    X_train_df = pd.DataFrame(xgb_data["X_train"], columns=features)
    X_val_df = pd.DataFrame(xgb_data["X_val"], columns=features)
    X_test_df = pd.DataFrame(xgb_data["X_test"], columns=features)

    # Initial training
    xgb_model.fit(
        X_train_df,
        xgb_data["y_train"],
        eval_set=[(X_val_df, xgb_data["y_val"])],
        verbose=False,
    )
    print("XGBoost training finished!")

    # Get feature importance
    importances = xgb_model.feature_importances_
    feature_importance_pairs = list(zip(features, importances))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

    n_features = len(features)
    n_keep = max(1, int(n_features * (100 - drop_percent) / 100))

    top_features = [pair[0] for pair in feature_importance_pairs[:n_keep]]
    dropped_features = [pair[0] for pair in feature_importance_pairs[n_keep:]]

    # Retrain with only top features
    # Filter data to keep only top features
    feature_indices = [features.index(f) for f in top_features]

    X_train_filtered = xgb_data["X_train"][:, feature_indices]
    X_val_filtered = xgb_data["X_val"][:, feature_indices]
    X_test_filtered = xgb_data["X_test"][:, feature_indices]

    # Create new model (fresh training)
    from src.models.xgboost_model import get_xgboost_model

    xgb_model_filtered = get_xgboost_model()

    # Train on filtered data
    X_train_filtered_df = pd.DataFrame(X_train_filtered, columns=top_features)
    X_val_filtered_df = pd.DataFrame(X_val_filtered, columns=top_features)

    xgb_model_filtered.fit(
        X_train_filtered_df,
        xgb_data["y_train"],
        eval_set=[(X_val_filtered_df, xgb_data["y_val"])],
        verbose=False,
    )

    print("XGBoost re-training finished!")

    # Evaluate
    preds = xgb_model_filtered.predict(X_test_filtered)
    probs = xgb_model_filtered.predict_proba(X_test_filtered)[:, 1]

    results = {
        # ...
    }
    return results