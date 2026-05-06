#import "../headers.typ": *
= Kamień Milowy 4
== Wprowadzone zmiany
=== Zmiana samplingu danych

w `get_dataloaders.py` zamiast shuffle=True zastosowano WeightedRandomSampler

#{
  set text(size: 10pt) 
  raw(read("../snippets/weighted_avg.py"), lang:"python")
}

*Dlaczego:*

 - W danych o churnie mamy *silnie niezbalansowane* klasy - ok. 73% osób pozostaje, tylko ok. 27% osób odchodzi (Wykres - @class-balance).

  - `shuffle=True` losuje próbki losowo, ale nie zmienia rozkładu klas - nadal w każdej epoce model widzi *dużo więcej przykładów klasy dominującej*.

  - WeightedRandomSampler nadaje wagi próbkom *odwrotnie proporcjonalnie* do liczebności ich klasy. Dzięki temu w każdej epoce model widzi tyle samo przykładów z klasy mniejszościowej, co z większościowej.

 - W efekcie model powinien nie ignorować klientów odchodzących, skutkując poprawieniem metryk
 
=== Zmniejszenie progu funkcji aktywacji `ChurnNeuralNet`
Próg został zmniejszony z `0.5` do `0.4`.

```python
        nn_results = evaluate_pytorch(nn_model, test_loader, DEVICE, threshold=0.4)
```

*Dlaczego:*

 - Domyślny 0.5 działa dobrze z założeniem, że klasy są tak samo prawdopodobne - co w naszym przypadku nie jest prawdą.

  - Dla niezbalansowanych danych, model często przewiduje niskie prawdopodobieństwo dla klasy mniejszościowej (np. 0.3). Przy progu 0.5 - żadna taka próbka nie zostanie zaklasyfikowana jako “odejście”.

  - Obniżenie progu do 0.4 zwiększa czułość (recall), czyli wykrywalność klientów, którzy *faktycznie odeszli*. Kosztem może być nieznaczny wzrost fałszywie pozytywnych, jednakże w analizowanym przez nas przypadku jest to akceptowalne - *najważniejsze jest, aby nie przegapić klientów którzy faktycznie odejdą.*

=== Zastąpienie `BCEWithLogitsLoss` przez `FocalLoss`

W pliku `trainer_pytorch.py` jako kryterium do trenowana zmieniono `BCEWithLogitsLoss` na `FocalLoss`

```py
  criterion = FocalLoss()
  # ...
  
  # ...
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)
```


*Dlaczego:*

 - Metoda Focal Loss została zaproponowana do detekcji obiektów w *silnie niezbalansowanych zbiorach*.

  - Redukuje wagę łatwych przykładów (gdzie model jest bardzo pewny klasy większościowej) i skupia się na trudnych, źle sklasyfikowanych przykładach (często należących do klasy mniejszościowej).

 - Model uczy się lepiej rozpoznawać rzadkie, ale ważne przypadki churnu, nawet jeśli są one trudniejsze do odróżnienia.

=== XGBoost - Opuszczenie najmniej ważnych cech

XGBoost trenuje model, następnie odrzuca 30% najmniej ważnych cech, po czym następnie trenuje nowy model (pomijając odrzucone cechy).

#{
  set text(size: 10pt) 
  raw(read("../snippets/xgboost_feature_selection.py"), lang:"python")
}


*Dlaczego:*

 - XGBoost ma wbudowaną regularyzację, ale zbędne cechy mogą:

  - wprowadzać szum,

  - powodować lekkie przetrenowanie,

  - wydłużać czas treningu.

 - Usunięcie najmniej ważnych cech upraszcza model, zmniejsza wariancję i może poprawiać generalizację.

 - Po ponownym trenowaniu często okazuje się, że model na mniejszej liczbie cech działa równie dobrze lub lepiej.