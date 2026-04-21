import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import RAW_DATA_PATH
from src.dataset import load_and_clean_data


def main():
    # Load data
    df = load_and_clean_data(RAW_DATA_PATH)

    # set up save dir for later
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots"
    )
    os.makedirs(save_dir, exist_ok=True)

    # initial cleanup
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # basic info
    print("--- BASIC INFORMATION ---")
    print(f"Number of records: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nTypes and missing data:\n", df.info())

    print("\n")
    print(df.describe())

    # target variable analysis
    print("\n--- CHURN STATISTICS ---")

    churn_counts = df["Churn"].value_counts(normalize=True) * 100

    churn_yes = churn_counts.get(1.0, 0)
    churn_no = churn_counts.get(0.0, 0)

    print(f"% of clients that left (1): {churn_yes:.2f}%")
    print(f"% of clients that stayed (0): {churn_no:.2f}%")

    # visualization
    sns.set_style("whitegrid")
    plt.figure()#figsize=(18, 12))

    # plt.subplot(2, 2, 1)
    # df["Churn"].value_counts().plot.pie(
    #     autopct="%1.1f%%", colors=["#66b3ff", "#ff9999"], startangle=90
    # )
    # plt.title("Churn pie plot")

    # plt.subplot(2, 2, 2)
    sns.histplot(
        data=df, x="MonthlyCharges", hue="Churn", multiple="stack", palette="magma"
    )
    plt.title("Monthly Charges vs Churn")

    # plt.subplot(2, 2, 3)
    # sns.boxplot(x="Churn", y="tenure", data=df, palette="Set2")
    # plt.title("Tenure vs Churn")

    # plt.subplot(2, 2, 4)
    # numeric_df = df.select_dtypes(include=[np.number])
    # sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    # plt.title("Correlation heatmap")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "5_charges_vs_churn.png"))
    plt.close()


if __name__ == "__main__":
    main()
