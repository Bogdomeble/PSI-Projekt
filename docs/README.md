## Kamień Milowy 1

 

### Problem

 Problem polega na przewidywaniu czasu, po którym dany klient zamierza zakończyć umowę z firmą zajmującą się telekomunikacją. Chcemy ten czas jak najbardziej przedłużyć w celu maksymalizacji dochodów pochodzących ze świadczenia usług internetowych (telewizja, internet światłowodowy itp.) 

### Cel Projektu

 Efektywna kampania marketingowa (predykcja za pomocą algorytmu sztucznej inteligencji), mająca na celu proponowanie klientom lepsze warunki umowy / promocje w przewidzianym okresie, w którym dany klient ma duże prawdopodobieństwo na odstąpienie od niej. 

### Wymagania:

- Zestaw danych dotyczący umów klientów, w szczególności zawierający dane o okresie czasu, po którym klient rozwiązał umowę _[(LINK - Wersja 2017)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)_ _[(Wersja 2019)](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)_
- Skalowalność: Wzrost rozmiaru danych wejściowych w realistycznych proporcjach nie powinien zwiększać gwałtownie czasu wykonania modelu.
- Bezpieczeństwo: Model nie powinien ujawniać w żaden sposób danych wrażliwych pochodzących od klientów.

<p align="center"><figure class="figure"><p><img src="images/kamienie.png" alt="" /></p>
</figure></p>

## Kamień Milowy 2

 Wybrany zbiór danych przedstawia bazę klientów fikcyjnej firmy telekomunikacyjnej. Głównym celem pracy z tymi danymi jest przewidywanie, który klient zrezygnuje z usług firmy / po jakim czasie zakończy on umowę z firmą (czyli przewidywanie zjawiska Churn). Aby łatwiej było go zrozumieć, te 21 kolumn można podzielić na 4 logiczne kategorie:

1. Dane demograficzne (Kim jest klient?)
   - CustomerID: Unikalny identyfikator klienta (np. 7590-VHVEG). Z punktu widzenia analityki ignoruje się go, bo nie niesie żadnej wiedzy.
   - gender: Płeć klienta (Male - Mężczyzna, Female - Kobieta).
   - SeniorCitizen: Czy klient jest seniorem? Wartość liczbowa: 1 oznacza Tak, 0 oznacza Nie.
   - Partner: Czy klient ma małżonka/partnera? (Yes/No).
   - Dependents: Czy klient ma osoby na utrzymaniu (np. dzieci)? (Yes/No).

1. Wykupione usługi (Z czego korzysta klient?)
   - PhoneService: Czy klient ma telefon stacjonarny? (Yes/No).
   - MultipleLines: Czy klient ma wiele linii telefonicznych? (Yes/No/No phone service).
   - InternetService: Rodzaj łącza internetowego. Może to być DSL (standardowe łącze), Fiber optic (szybki światłowód) lub No (brak internetu).
   - Dodatkowe usługi internetowe (Jeśli klient ma internet, może dobrać te opcje - przyjmują wartości Yes / No / No internet service):
   - OnlineSecurity: Dodatkowe zabezpieczenia sieciowe.
   - OnlineBackup: Usługa kopii zapasowej w chmurze.
   - DeviceProtection: Ubezpieczenie/ochrona sprzętu (np. routera).
   - TechSupport: Priorytetowe wsparcie techniczne.
   - StreamingTV: Usługa telewizji przez internet.
   - StreamingMovies: Usługa VOD (filmy na życzenie).

1. Informacje o koncie i płatnościach (Jak klient płaci?)
   - Tenure (Staż): Bardzo ważna kolumna. Liczba miesięcy, przez które klient jest z firmą (np. 1 oznacza nowego klienta, 45 to lojalny klient od prawie 4 lat).
   - Contract: Rodzaj umowy:
   - Month-to-month (z miesiąca na miesiąc - ci klienci najszybciej odchodzą),
   - One year (umowa na rok),
   - Two year (umowa na dwa lata).
   - PaperlessBilling: Czy klient zrezygnował z rachunków papierowych na rzecz elektronicznych? (Yes/No).
   - PaymentMethod: Metoda płatności (np. Electronic check - czek elektroniczny, Mailed check - czek pocztowy, Bank transfer - przelew, Credit card - karta kredytowa).
   - MonthlyCharges: Miesięczna kwota rachunku w dolarach (np. $29.85).
   - TotalCharges: Suma wszystkich opłat, jakie klient wniósł od początku trwania umowy (w przybliżeniu tenure \* MonthlyCharges). To właśnie tę kolumnę musimy modyfikować, bo u zupełnie nowych klientów (tenure = 0) to pole było w pliku puste (spacja) zamiast zera.
2. Zmienna docelowa (Cel analizy)

Churn: odpowiada na pytanie: „Czy w ciągu ostatniego miesiąca ten klient zrezygnował z usług firmy?”

- Yes - Klient odszedł (Strata dla firmy).
- No - Klient został.

<p align="center"><figure class="figure"><p><img src="images/data_sample.png" alt="" /></p>
</figure></p>

Przeczyszczone dane są następnie przekazywane do funkcji `prepare_data_splits()`:

`def prepare_data_splits(df):
    X = df.drop(columns=[TARGET_COLUMN]).values
    y = df[TARGET_COLUMN].values

    # Divide the data into train,test,validate subsets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    # scale data for NNs
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)`

Funkcja dzieli dane na trzy zbiory: zbiór treningowy, zbiór walidacyjny i zbiór testowy.

`class ChurnDataset(Dataset):
    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)

        # For binary classification pytorch need someithing like (n,1) instead of just (n)

        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders():

    df = load_and_clean_data(RAW_DATA_PATH)

    feature_names = df.drop(columns=[TARGET_COLUMN]).columns.tolist()

    #save modified data to temporaty file

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data_splits(df)

    # Dataloaders for python

    train_loader = DataLoader(ChurnDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
    """drop_last=True fixes the Expected more than 1 value per channel when training, got input size torch.Size([1, 64])"""
    val_loader = DataLoader(ChurnDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ChurnDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # XGboost dictionary for the selected training sets
    xgb_data = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names  # <--- this is for the xgboost plot file
    }

    # we return input_dims here
    input_dim = X_train.shape[1]
    return train_loader, val_loader, test_loader, xgb_data, input_dim
`

Podzielone dane następnie są owijane w DataLoader (do pytorcha) czy zwracane w słowniku `xgb_data` (do xgboosta) 

### Wykresy

<p align="center"><figure class="figure"><p><img src="images/1_class_balance.png" alt="" /></p>
</figure></p>

<p align="center"><figure class="figure"><p><img src="images/2_monthly_charges_boxplot.png" alt="" /></p>
</figure></p>

<p align="center"><figure class="figure"><p><img src="images/3_contract_vs_churn.png" alt="" /></p>
</figure></p>

<p align="center"><figure class="figure"><p><img src="images/4_correlation_heatmap.png" alt="" /></p>
</figure></p>

<p align="center"><figure class="figure"><p><img src="images/5_charges_vs_churn.png" alt="" /></p>
</figure></p>

1. Wykres słupkowy (Distribution of Churn - Rysunek 3) - Balans klas:
   Co widać: 5174 klientów pozostało w firmie, a 1869 zrezygnowało. Wniosek: Rezygnację (churn) dotyczą około 26 - 27% klientów. Zbiór danych jest „niezbalansowany” (więcej lojalnych niż odchodzących), co później wymusi zmianę wag tych zmiennych w kodzie dla sieci XGBoost.
   1. Wykres pudełkowy (Monthly charges vs churn - Rysunek 4) - Opłaty miesięczne:Co widać: Pudełko dla odchodzących (Yes - czerwone) leży zauważalnie wyżej niż dla zostających (No - zielone). Wniosek: Klienci, którzy rezygnują z usług, mają zazwyczaj wyższe miesięczne rachunki (mediana ok. 80), niż ci którzy zostają (mediana ok. 65). Wyższe koszty sprzyjają ucieczce do konkurencji.

1. Pogrupowane słupki (Contract type vs churn risk - Rysunek 5) - Rodzaj umowy
   Co widać: Ogromna czerwona kolumna (Yes) przy umowach „Month-to-month”. Prawie brak czerwieni przy umowach na 2 lata. Wniosek: Klienci bez długoterminowego zobowiązania (z miesiąca na miesiąc) odchodzą masowo. Podpisanie umowy na rok lub dwa drastycznie zmniejsza ryzyko utraty klienta.

1. Mapa cieplna (Correlation matrix - Rysunek 6) - Korelacje liczbowe
   Co widać: Współczynniki zależności. Na przykład na przecięciu tenure i Churn\_Num mamy współczynnik korelacji równy −0.35. Wniosek: Istnieje ujemna korelacja między stażem a odejściem. Oznacza to, że im dłużej ktoś jest z firmą (wyższe tenure), tym mniejsze prawdopodobieństwo, że zrezygnuje.

1. Histogram ze stosami (Monthly Charges vs Churn - Rysunek 7) Co widać: “Duża fioletowa wieża” (zostający przy kwocie $20) oraz “szeroka różowa górka” (odchodzący w przedziale $70-$105). Wniosek: Uzupełnia to wykres pudełkowy. Firma ma ogromną rzeszę lojalnych klientów, którzy płacą bardzo mało (prawdopodobnie utrzymują najtańszą, podstawową usługę). Problem rezygnacji dotyka głównie klientów o średnich i wysokich rachunkach.
