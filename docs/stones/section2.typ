#import "../headers.typ": *
= Kamień Milowy 2
Wybrany zbiór danych przedstawia bazę klientów fikcyjnej firmy telekomunikacyjnej. Głównym celem pracy z tymi danymi jest przewidywanie, który klient zrezygnuje z usług firmy / po jakim czasie zakończy on umowę z firmą (czyli przewidywanie zjawiska Churn). Aby łatwiej było go zrozumieć, te 21 kolumn można podzielić na 4 logiczne kategorie:


+ Dane demograficzne (Kim jest klient?)

  - CustomerID: Unikalny identyfikator klienta (np. 7590-VHVEG). Z punktu widzenia analityki ignoruje się go, bo nie niesie żadnej wiedzy.
  - gender: Płeć klienta (Male - Mężczyzna, Female - Kobieta).
  - SeniorCitizen: Czy klient jest seniorem? Wartość liczbowa: 1 oznacza Tak, 0 oznacza Nie.
  - Partner: Czy klient ma małżonka/partnera? (Yes/No).
  - Dependents: Czy klient ma osoby na utrzymaniu (np. dzieci)? (Yes/No).


+ Wykupione usługi (Z czego korzysta klient?)

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

+ Informacje o koncie i płatnościach (Jak klient płaci?)

  - Tenure (Staż): Bardzo ważna kolumna. Liczba miesięcy, przez które klient jest z firmą (np. 1 oznacza nowego klienta, 45 to lojalny klient od prawie 4 lat).
  - Contract: Rodzaj umowy:
  - Month-to-month (z miesiąca na miesiąc - ci klienci najszybciej odchodzą),
  - One year (umowa na rok),
  - Two year (umowa na dwa lata).
  - PaperlessBilling: Czy klient zrezygnował z rachunków papierowych na rzecz elektronicznych? (Yes/No).
  - PaymentMethod: Metoda płatności (np. Electronic check - czek elektroniczny, Mailed check - czek pocztowy, Bank transfer - przelew, Credit card - karta kredytowa).
  - MonthlyCharges: Miesięczna kwota rachunku w dolarach (np. \$29.85).
  - TotalCharges: Suma wszystkich opłat, jakie klient wniósł od początku trwania umowy (w przybliżeniu tenure \* MonthlyCharges). To właśnie tę kolumnę musimy modyfikować, bo u zupełnie nowych klientów (tenure = 0) to pole było w pliku puste (spacja) zamiast zera.
+  Zmienna docelowa (Cel analizy)

Churn: odpowiada na pytanie: "Czy w ciągu ostatniego miesiąca ten klient zrezygnował z usług firmy?"
  - Yes - Klient odszedł (Strata dla firmy).
  - No - Klient został.

#figure(
  image("../images/data_sample.png"),
  caption: [Wybrane dane i nagłówki kolumn ze zbioru danych]
) <data-sample>
#columns(2)[
  Za czyszczenie danych w naszym kodzie odpowiada funkcja `load_and_clean_data()`. Załączamy kod funkcji; \
  Funkcja ta:
    - Wczytuje .csv naszych danych
    - Usuwa kolumnę `customerID`, która jest zbędna na potrzeby naszej analizy
    - Zamienia wszystkie puste pola (“ “) na NaN, które następnie są zmieniane na `0`
    - Mapuje kolumnę `Churn`  z “Yes” i “No” na odpowiednio 1 i 0
    - Mapuje wartości na zmienne liczbowe - one-hot encoding
    - Castuje wszystkie wartości na typ zmiennoprzecinkowy
  #colbreak()
  #{
    // 11pt jest dużo line wrapping
    // 10pt to chyba taki OK balans
    set text(size: 10pt) 
    raw(read("../snippets/load_and_clean_data.py"), lang:"python")
  }
]
Przeczyszczone dane są następnie przekazywane do funkcji `prepare_data_splits()`:

#{
  set text(size: 10pt) 
  raw(read("../snippets/prepare_data_splits.py"), lang:"python")
}

Funkcja dzieli dane na trzy zbiory: zbiór treningowy, zbiór walidacyjny i zbiór testowy.

#{
  set text(size: 10pt) 
  raw(read("../snippets/get_dataloaders.py"), lang:"python")
}

Podzielone dane następnie są owijane w DataLoader (do pytorcha) czy zwracane w słowniku `xgb_data` (do xgboosta)
== Wykresy

#figure(
  image("../images/1_class_balance.png"),
  caption: [Wykres słupkowy zmiennej `Churn`]
) <class-balance>

#figure(
  image("../images/2_monthly_charges_boxplot.png"),
  caption: [Boxplot opłat miesięcznych]
) <monthly-charges>

#figure(
  image("../images/3_contract_vs_churn.png"),
  caption: [Churn w zależności od typu kontratu]
) <contract-vs-churn>

#figure(
  image("../images/4_correlation_heatmap.png"),
  caption: [Heatmap zależności wybranych zmiennych]
) <correlation-heatmap>

#figure(
  image("../images/5_charges_vs_churn.png", width: 80%),
  caption: [Zależność między opłatami miesięcznymi a zmienną `Churn` - Histogram]
) <charges-vs-churn>

+ Wykres słupkowy (Distribution of Churn - @class-balance) - Balans klas:

  Co widać: 5174 klientów pozostało w firmie, a 1869 zrezygnowało.
  Wniosek: Rezygnację (churn) dotyczą około 26 - 27% klientów. Zbiór danych jest "niezbalansowany" (więcej lojalnych niż odchodzących), co później wymusi zmianę wag tych zmiennych w kodzie dla sieci XGBoost.

  + Wykres pudełkowy (Monthly charges vs churn - @monthly-charges) - Opłaty miesięczne:

  Co widać: Pudełko dla odchodzących (Yes - czerwone) leży zauważalnie wyżej niż dla zostających (No - zielone).
  Wniosek: Klienci, którzy rezygnują z usług, mają zazwyczaj wyższe miesięczne rachunki (mediana ok. 80), niż ci którzy zostają (mediana ok. 65). Wyższe koszty sprzyjają ucieczce do konkurencji.

+ Pogrupowane słupki (Contract type vs churn risk - @contract-vs-churn) - Rodzaj umowy

  Co widać: Ogromna czerwona kolumna (Yes) przy umowach "Month-to-month". Prawie brak czerwieni przy umowach na 2 lata.
  Wniosek: Klienci bez długoterminowego zobowiązania (z miesiąca na miesiąc) odchodzą masowo. Podpisanie umowy na rok lub dwa drastycznie zmniejsza ryzyko utraty klienta.

+ Mapa cieplna (Correlation matrix - @correlation-heatmap) - Korelacje liczbowe

  Co widać: Współczynniki zależności. Na przykład na przecięciu tenure i Churn_Num mamy współczynnik korelacji równy -0.35.
  Wniosek: Istnieje ujemna korelacja między stażem a odejściem. Oznacza to, że im dłużej ktoś jest z firmą (wyższe tenure), tym mniejsze prawdopodobieństwo, że zrezygnuje. 

+ Histogram ze stosami (Monthly Charges vs Churn - @charges-vs-churn)
  Co widać: “Duża fioletowa wieża” (zostający przy kwocie \$20) oraz “szeroka różowa górka” (odchodzący w przedziale \$70-\$105).
  Wniosek: Uzupełnia to wykres pudełkowy. Firma ma ogromną rzeszę lojalnych klientów, którzy płacą bardzo mało (prawdopodobnie utrzymują najtańszą, podstawową usługę). Problem rezygnacji dotyka głównie klientów o średnich i wysokich rachunkach.
