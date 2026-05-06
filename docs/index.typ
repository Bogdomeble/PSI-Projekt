// Kamienie milowe w oddzielnych plikach później
// w trakcie pisania nowego można tutaj i potem ctrl+x ctrl+v
// #set text(lang: "pl")
#import "template.typ": *
#show heading: smallcaps
#show link: set text(blue)
#show link: emph

#let x-target = sys.inputs.at("x-target", default: "pdf")

#let temat = "Przewidywanie okresu wypowiedzenia umowy na podstawie danych klientów w celu zwiększenia zysku ze świadczenia usług"

#if(x-target != "md") {
  show: project.with(
    title: [Podstawy Sztucznej Inteligencji],
    subtitle: [#temat],
    authors: (
      (name: "Bartosz Bodulski", affiliation: "Informatyka Techniczna"),
      (name: "Maciej Gładysiak", affiliation: "Informatyka Techniczna"),
    ),
    date: "22 Kwietnia 2026",
  )
[
  #set text(lang: "pl")
  #outline()
]

}

#pagebreak()

#set text(lang: "pl")
#include "stones/section1.typ"
#pagebreak() // nowa strona
#include "stones/section2.typ"
#pagebreak()
#include "stones/section3.typ"
#pagebreak()
#include "stones/section4.typ"