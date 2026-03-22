
***

# Przewidywanie Odejścia Klienta (Telco Customer Churn)

Projekt z zakresu Sztucznej Inteligencji zrealizowany w języku Python. 
Celem projektu jest zbudowanie i porównanie dwóch modeli uczenia maszynowego: **Sztucznej Sieci Neuronowej (PyTorch)** oraz algorytmu opartego na drzewach decyzyjnych **(XGBoost)**, w celu klasyfikacji, czy klient telekomunikacyjny zrezygnuje z usług firmy (zjawisko tzw. *Churnu*).

## Wykorzystane technologie
* **Język:** Python 3.9+
* **Uczenie maszynowe:** PyTorch (ze wsparciem dla CUDA), XGBoost, Scikit-Learn
* **Przetwarzanie danych:** Pandas, NumPy

## Struktura Projektu
```text
PSI-Projekt/
├── data/
│   ├── processed/          # Wygenerowane, oczyszczone dane (tworzone automatycznie)
│   └── raw/                # Surowe dane (tutaj umieść plik z Kaggle)
├── src/                    # Główny kod źródłowy projektu
│   ├── core/               # Skrypty trenujące modele
│   ├── models/             # Architektury modeli (PyTorch & XGBoost)
│   ├── config.py           # Główna konfiguracja i parametry uczenia
│   ├── dataset.py          # Data Pipeline (czyszczenie, split, DataLoader)
│   └── main.py             # Główny punkt wejścia do aplikacji
├── requirements.txt        # Lista zależności i bibliotek
└── README.md               # Dokumentacja projektu
```

## Konfiguracja i uruchomienie (Setup krok po kroku)

### 1. Pobranie repozytorium
Skopiuj projekt na swój dysk lokalny i przejdź do głównego folderu:
```bash
git clone https://github.com/Bogdomeble/PSI-Projekt.git
cd PSI-Projekt
```

### 2. Utworzenie wirtualnego środowiska
Zaleca się uruchamianie projektu w wyizolowanym środowisku.
```bash
python3 -m venv venv
```
Aktywacja środowiska:
* **Linux / macOS:** `source venv/bin/activate`
* **Windows:** `venv\Scripts\activate`

### 3. Instalacja zależności
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Uwaga: Instalacja PyTorcha domyślnie uwzględnia pakiety wspierające akcelerację kart graficznych Nvidia (CUDA).*

### 4. Przygotowanie danych
1. Pobierz zbiór danych Telco Customer Churn z [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Pobrany plik nazwij `Telco-Customer-Churn.csv` i umieść go w katalogu `data/raw/`.

### 5. Uruchomienie projektu
Aby uniknąć problemów ze ścieżkami (PYTHONPATH), projekt należy uruchamiać jako **moduł**, będąc w głównym katalogu (`PSI-Projekt/`):

```bash
python3 -m src.main
```

Jeśli konfiguracja przebiegła pomyślnie, w konsoli pojawi się informacja o używanym urządzeniu (CPU/CUDA) oraz wymiary przygotowanych zbiorów danych gotowych do treningu modeli.
