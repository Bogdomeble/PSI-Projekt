import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import *
from src.dataset import load_and_clean_data

def main():
    # 1. Wczytanie danych
    df = load_and_clean_data(RAW_DATA_PATH)

    # --- Wstępne czyszczenie ---
    # TotalCharges jest wczytywane jako object, musimy zamienić na numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True) # Usuwamy puste rekordy powstałe przy konwersji

    # 2. ANALIZA STATYSTYCZNA I OPISOWA
    print("--- PODSTAWOWE INFORMACJE ---")
    print(f"Liczba rekordów: {df.shape[0]}")
    print(f"Liczba kolumn: {df.shape[1]}")
    print("\nTypy danych i braki:\n", df.info())

    print("\n--- STATYSTYKI OPISOWE (Zmienne numeryczne) ---")
    print(df.describe())

    # 3. ANALIZA PROPORCJI (Target Variable)
    print("\n--- ROZKŁAD REZYGNACJI (CHURN) ---")

    churn_counts = df['Churn'].value_counts(normalize=True) * 100

    churn_yes = churn_counts.get(1.0, 0)
    churn_no = churn_counts.get(0.0, 0)

    print(f"Procent klientów, którzy odeszli (1): {churn_yes:.2f}%")
    print(f"Procent klientów, którzy zostali (0): {churn_no:.2f}%")

    # 4. WIZUALIZACJA I ZALEŻNOŚCI
    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 12))

    # A. Rozkład Churn (Wykres kołowy)
    plt.subplot(2, 2, 1)
    df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90)
    plt.title('Proporcja rezygnacji (Churn)')

    # B. Miesięczne opłaty vs Churn (Histogram)
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple="stack", palette='magma')
    plt.title('Miesięczne opłaty a rezygnacja')

    # C. Staż klienta (Tenure) vs Churn (Boxplot)
    plt.subplot(2, 2, 3)
    sns.boxplot(x='Churn', y='tenure', data=df, palette='Set2')
    plt.title('Staż klienta (w miesiącach) a rezygnacja')

    # D. Macierz korelacji (tylko dla cech numerycznych)
    plt.subplot(2, 2, 4)
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
    plt.title('Korelacje między zmiennymi numerycznymi')

    plt.tight_layout()
    plt.show()

    # 5. ANALIZA ZMIENNYCH KATEGORIALNYCH (Segmentacja)
    # Sprawdzamy wpływ rodzaju umowy na odejście
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='viridis')
    plt.title('Wpływ rodzaju umowy na rezygnację')
    plt.show()

    print("\n--- ANALIZA ZAKOŃCZONA ---")


if __name__ == "__main__":
    main()
