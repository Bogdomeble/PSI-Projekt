import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

# We import our plotting tools
from src.utils.plotting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_xgboost_importance,
)


def train_and_eval_xgboost(xgb_model, xgb_data):
    print("\n--- Starting XGBoost Training ---")
    features = xgb_data["feature_names"]
    X_train_df = pd.DataFrame(xgb_data["X_train"], columns=features)
    X_val_df = pd.DataFrame(xgb_data["X_val"], columns=features)
    X_test_df = pd.DataFrame(xgb_data["X_test"], columns=features)

    xgb_model.fit(
        X_train_df,
        xgb_data["y_train"],
        eval_set=[(X_val_df, xgb_data["y_val"])],
        verbose=False,
    )
    print("XGBoost training finished!")

    # Predictions and Probabilities
    preds = xgb_model.predict(xgb_data["X_test"])
    probs = xgb_model.predict_proba(xgb_data["X_test"])[
        :, 1
    ]  # Get probability for Churn class (1)

    # Generate all plots for XGBoost
    plot_confusion_matrix(xgb_data["y_test"], preds, model_name="xgboost")
    plot_roc_curve(xgb_data["y_test"], probs, model_name="xgboost")
    plot_xgboost_importance(xgb_model)

    print(
        "Plots for models (ROC, Confusion Matrix, Feature Importance) have been saved in the 'plots/' folder."
    )

    return {
        "Accuracy": accuracy_score(xgb_data["y_test"], preds),
        "F1-Score": f1_score(xgb_data["y_test"], preds),
        "Recall": recall_score(xgb_data["y_test"], preds),
        "ROC-AUC": roc_auc_score(xgb_data["y_test"], probs),
    }


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

    # Calculate how many features to keep
    n_features = len(features)
    n_keep = max(1, int(n_features * (100 - drop_percent) / 100))

    # Get top features
    top_features = [pair[0] for pair in feature_importance_pairs[:n_keep]]
    dropped_features = [pair[0] for pair in feature_importance_pairs[n_keep:]]

    if verbose:
        print("\nFeature importance ranking:")
        for i, (feature, imp) in enumerate(feature_importance_pairs[:10], 1):
            print(f"      {i}. {feature}: {imp:.4f}")

        print(f"\nKeeping top {100 - drop_percent}% ({n_keep}/{n_features} features)")

        if dropped_features:
            print(f"\nDropping {len(dropped_features)} low-importance features:")
            for feature in dropped_features[:5]:
                print(f"      - {feature}")
            if len(dropped_features) > 5:
                print(f"      ... and {len(dropped_features) - 5} more")

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

    # Generate plots
    plot_confusion_matrix(xgb_data["y_test"], preds, model_name="xgboost_filtered")
    plot_roc_curve(xgb_data["y_test"], probs, model_name="xgboost_filtered")
    plot_xgboost_importance(xgb_model_filtered, model_name="xgboost_filtered")

    # Compare with original (optional)
    original_preds = xgb_model.predict(X_test_df)
    original_probs = xgb_model.predict_proba(X_test_df)[:, 1]

    results = {
        "Accuracy": accuracy_score(xgb_data["y_test"], preds),
        "F1-Score": f1_score(xgb_data["y_test"], preds),
        "Recall": recall_score(xgb_data["y_test"], preds),
        "ROC-AUC": roc_auc_score(xgb_data["y_test"], probs),
        "original_accuracy": accuracy_score(xgb_data["y_test"], original_preds),
        "features_used": n_keep,
        "features_dropped": len(dropped_features),
        "top_features": top_features,
    }
    if verbose:
        print("\nPerformance Comparison:")
        print(f"   Original Accuracy: {results['original_accuracy']:.4f}")
        print(f"   Filtered Accuracy: {results['Accuracy']:.4f}")
        improvement = results["Accuracy"] - results["original_accuracy"]
        print(f"   {'Improvement' if improvement > 0 else 'Slight decrease'}: {improvement:+.4f}")

    return results
