import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import RAW_DATA_PATH

# style
sns.set_theme(style="whitegrid")

def load_data_for_eda():
    """Loads data and fixes only the TotalCharges column for visualization."""
    df = pd.read_csv(RAW_DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

def plot_class_balance(df, save_dir):
    """Plot 1: Class balance - how many customers left and how many stayed."""
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x='Churn', palette=['#2ecc71', '#e74c3c'])
    plt.title('Distribution of the target variable (Churn)')
    plt.xlabel('Did the customer leave? (Churn)')
    plt.ylabel('Number of customers')

    # Add values above the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_class_balance.png'))
    plt.close()

def plot_charges_vs_churn(df, save_dir):
    """Plot 2: Monthly charges vs customer churn (Boxplot)."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette=['#2ecc71', '#e74c3c'])
    plt.title('Monthly charges vs churn decision')
    plt.xlabel('Did the customer leave?')
    plt.ylabel('Monthly charges ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_monthly_charges_boxplot.png'))
    plt.close()

def plot_contract_vs_churn(df, save_dir):
    """Plot 3: Contract type vs customer churn."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette=['#2ecc71', '#e74c3c'])
    plt.title('Contract type vs churn risk')
    plt.xlabel('Contract type')
    plt.ylabel('Number of customers')
    plt.legend(title='Churn')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_contract_vs_churn.png'))
    plt.close()

def plot_correlation_heatmap(df, save_dir):
    """Plot 4: Correlation matrix for numerical variables."""
    # Select only numeric columns + convert Churn to numbers just for this plot
    df_numeric = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
    df_numeric['Churn_Num'] = df['Churn'].map({'Yes': 1, 'No': 0})

    corr_matrix = df_numeric.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation matrix of numerical features')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_correlation_heatmap.png'))
    plt.close()

def main():
    print("Starting to generate plots...")
    df = load_data_for_eda()

    # Create a folder for plots in the main project directory
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Generate plots
    plot_class_balance(df, save_dir)
    plot_charges_vs_churn(df, save_dir)
    plot_contract_vs_churn(df, save_dir)
    plot_correlation_heatmap(df, save_dir)

    print(f"Success! The plots have been generated and saved in the folder: {save_dir}/")

if __name__ == "__main__":
    main()
