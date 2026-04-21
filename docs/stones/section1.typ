#import "../headers.typ": *
= Kamień Milowy 1
== Problem
Problem polega na przewidywaniu czasu, po którym dany klient zamierza zakończyć umowę z firmą zajmującą się telekomunikacją. Chcemy ten czas jak najbardziej przedłużyć w celu maksymalizacji dochodów pochodzących ze świadczenia usług internetowych (telewizja, internet światłowodowy itp.)
== Cel Projektu
Efektywna kampania marketingowa (predykcja za pomocą algorytmu sztucznej inteligencji), mająca na celu proponowanie klientom lepsze warunki umowy / promocje w przewidzianym okresie, w którym dany klient ma duże prawdopodobieństwo na odstąpienie od niej.
== Wymagania:
- Zestaw danych dotyczący umów klientów, w szczególności zawierający dane o okresie czasu, po którym klient rozwiązał umowę #link("https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data")[(LINK - Wersja 2017)] #link("https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113")[(Wersja 2019)]
- Skalowalność: Wzrost rozmiaru danych wejściowych w realistycznych proporcjach nie powinien zwiększać gwałtownie czasu wykonania modelu.
- Bezpieczeństwo: Model nie powinien ujawniać w żaden sposób danych wrażliwych pochodzących od klientów.

// figure nie jest potrzebne samo w sobie
// ale pozwala dodać `caption`
// i odniesienie
#figure(
  image("../images/kamienie.png", width:74%),
  caption: [Wybrane przez nas kamienie milowe]
) <kamienie-milowe>
