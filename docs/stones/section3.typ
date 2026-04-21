#import "../headers.typ": *
= Kamień Milowy 3
== Opis
W ramach projektu zdecydowano się na podejście hybrydowe, implementując dwa niezależne modele, aby porównać skuteczność algorytmów drzewiastych z sieciami neuronowymi w zadaniu klasyfikacji binarnej (Churn: Tak/Nie).

Zaimplementowane modele:		
- Konfiguracja XGBClassifier (XGBoost).
- Klasa ChurnNeuralNet (PyTorch).

== Model 1: XGBoost (Extreme Gradient Boosting)
  
*Dlaczego:* Jest to obecnie jeden z najskuteczniejszych algorytmów dla danych tabelarycznych. Świetnie radzi sobie z brakującymi danymi i nieliniowymi zależnościami.
- *Kluczowe cechy implementacji:*
  - Zastosowano *skalowanie wag klas (scale_pos_weight=2.76)*, aby zaradzić nierównowadze w zbiorze danych (więcej osób zostaje w sieci, niż z niej odchodzi).
  - Ustawiono parametry zapobiegające overfittingowi: max_depth=4, subsample=0.8 oraz learning_rate=0.05.
- *Plik:* `src/models/xgboost_model.py`

#{
  set text(size: 10pt) 
  raw(read("../snippets/models/xgboost_model.py"), lang:"python")
}
#pagebreak()
== Model 2: Głęboka Sieć Neuronowa (PyTorch)
*Dlaczego:* Pozwala na wychwycenie bardzo złożonych, ukrytych korelacji między cechami, które mogą umknąć algorytmom drzewiastym.
- *Architektura (zaimplementowana w ChurnNeuralNet)*
  - *Warstwy:* Wejściowa -> 64 neurony -> 32 neurony -> Wyjściowa (1 neuron).
  - *Regularyzacja: *Zastosowano warstwy *Dropout* (30% i 20%) oraz *Batch Normalization*, aby model był stabilny i nie "uczył się na pamięć" danych treningowych.
  - *Funkcja aktywacji:* ReLU dla warstw ukrytych.
- *Plik:* `src/models/xgboost_model.py`

#{
  set text(size: 10pt) 
  raw(read("../snippets/models/neural_net.py"), lang:"python")
}

*Logika treningu (src/core/trainer.py)*
- Zaimplementowano ustandaryzowaną pętlę treningową z użyciem optymalizatora *Adam* oraz funkcji straty *BCEWithLogitsLoss* (idealna dla klasyfikacji binarnej w PyTorch).

*Oczekiwany wynik*

*Działające środowisko modelowania:* Stworzono modularną strukturę kodu, która pozwala na łatwą wymianę modeli i hiperparametrów.
== Wyniki
#figure(
  image("../images/model_shape.png"),
  caption: [Kształt modelu z torchinfo summary]
) <model-shape>

Modele są już zintegrowane z systemem metryk (Accuracy, F1-Score, Recall) w głównym skrypcie src/main.py. Wyniki dla obu modeli po podzieleniu datasetu  na sekcje 70/15/15 (trening/test/walidacja) wyglądają następująco:


#figure(
  image("../images/stone3_results.png"),
  caption: [Wyniki wytrenowanych modeli]
) <model-results>

#figure(
  image("../images/confusion_matrix_pytorch.png", width:90%),
  caption: [Confusion matrix dla pytorcha]
) <confusion-pytorch>

#figure(
  image("../images/confusion_matrix_xgboost.png", width:90%),
  caption: [Confusion matrix dla xgboosta]
) <confusion-xgboost>
#pagebreak()

=== Opis Metryk
*Accuracy (Dokładność)*
- *Co mówi:* Jaki procent wszystkich decyzji modelu był poprawny?
- *Wzór:* (Wszystkie dobre decyzje) / (Wszystkie przypadki)

Z danych wynika, że ponad 80% klientów nie kończy umowy z firmą. Model będzie wybierał sztywno tą opcję w celu uzyskania jak największej dokładności i może “okłamywać” resztę analizy. Tej mierze należy ufać tylko, gdy liczba klientów odchodzących i zostających jest mniej więcej równa.

*Recall (Czułość / Pełność)*
- *Co mówi:* Ile osób z tych, które naprawdę odeszły, nasz model zdołał poprawnie wskazać?
- *Wzór:* (Poprawnie wykryty Churn) / (Wszyscy, którzy faktycznie odeszli)

To jest prawdopodobnie najważniejsza metryka dla działu utrzymania klienta. Jeśli Recall wynosi 0.80, to znaczy, że zidentyfikowaliśmy 80% osób planujących odejście. Pozostałe 20% to "uciekinierzy", których nie zauważyliśmy i niestety straciliśmy. Chcemy, aby Recall był jak najwyższy, żeby nie przegapić żadnego klienta, którego można jeszcze uratować promocją.

*F1-Score*
- *Co mówi:* To średnia (harmoniczna) z Recall i Precision. Szuka balansu.
=== Ogólne Porównanie
Mimo że sieć neuronowa (PyTorch) ma wyższą ogólną dokładność, to w tym konkretnym zadaniu (przewidywanie odejścia klientów) zdecydowanym zwycięzcą jest XGBoost.
  - Accuracy (Dokładność):
    - PyTorch (79,75%) wypada lepiej niż XGBoost (75,40%).
    - Interpretacja: Sieć neuronowa rzadziej się myli ogółem, ale prawdopodobnie wynika to z tego, że jest bardzo "ostrożna" i najczęściej przewiduje, że klient zostanie (bo takich klientów jest w bazie najwięcej).

  - Recall (Czułość):
    - XGBoost: (82,14%) przewyższa PyTorch: (55%).
    - Interpretacja:  Tutaj jest najważniejsza różnica pomiędzy modelami. XGBoost poprawnie wykrywa aż 82% osób, które faktycznie chcą odejść. Sieć neuronowa wykrywa tylko 55% ,czyli prawie połowę uciekających klientów po prostu pomija. Tak wysoki wynik XGBoost to efekt zastosowania w kodzie parametru scale_pos_weight=2.76, który "zmusił" model do zwracania większej uwagi na klientów odchodzących.

  - F1-Score:
    - XGBoost (63,89%) przewyższa PyTorch (59%).
    - Interpretacja: F1-Score potwierdza, że choć XGBoost częściej się myli (niższe Accuracy), to jego ogólna wartość użytkowa w balansowaniu wykrywania zjawiska “Churnu” w tym zbiorze danych jest większa.