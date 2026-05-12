#import "../headers.typ": *
= Kamień Milowy 5
== Opis
W ramach wdrożenia modelu postanowiliśmy stworzyć frontend który pozwala użytkownikowi na wprowadzenie danych klienta na podstawie których model dokonuje predykcji i zwraca prawdopodobieństwo wystąpienia zjawiska "churn". Do frontend-u używamy biblioteki #link("https://streamlit.io/")[Streamlit], która pozwala w prosty sposób stworzyć interfejs do wizualizacji danych używając prostych formularzy i guzików.
=== Działanie aplikacji
Po uruchomieniu aplikacji ładowane są oba wyeksportowane wcześniej modele:

#{
  set text(size: 10pt) 
  raw(read("../snippets/app_load_models.py"), lang:"python")
}

i przy ich użyciu - po kliknięciu odpowiedniego przycisku w interfejsie użytkownika - dokonywana jest predykcja.
== Przykładowe przypadki
#figure(
  image("../images/app_example_one.png"),
  caption: [Przykład interfejsu aplikacji - dane losowego klienta z datasetu]
) <example-one>

#figure(
  image("../images/app_example_two.png"),
  caption: [Przykład interfejsu aplikacji - dane losowego klienta z datasetu]
) <example-two>

#figure(
  image("../images/app_example_three.png"),
  caption: [Przykład interfejsu aplikacji - dane losowego klienta z datasetu]
) <example-three>
#pagebreak()
=== Dodatek: poprawione wyniki dla zbioru testowego i walidacyjnego z kamienia milowego 4

#figure(
  image("../images/5_test.png"),
  caption: [Poprawione wyniki dla zbioru testowego]
) <fix-test>

#figure(
  image("../images/5_validation.png"),
  caption: [Poprawione wyniki dla zbioru walidacyjnego]
) <fix-validation>

Poprawione wyniki dla zbioru walidacyjnego oraz testowego pokazują bardzo wysoki (>= 0.9) recall dla PyTorcha, kosztem niższego (0.64 do 0.65) accuracy, co skutkuje niskim F1-Score. XGBoost cechuje się gorszym recall-em (0.82 na zbiorze testowym, 0.77 na walidacyjnym), ale za to ma większe accuracy (ok. 0.75), co skutkuje lepszym od PyTorcha F1-Score.