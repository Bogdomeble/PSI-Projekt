// Kamienie milowe w oddzielnych plikach później
// w trakcie pisania nowego można tutaj i potem ctrl+x ctrl+v
<<<<<<< HEAD
=======
// #set text(lang: "pl")
>>>>>>> ea2b078 (complete milestone 3)
#import "template.typ": *
#show heading: smallcaps
#show link: set text(blue)
#show link: emph

#let x-target = sys.inputs.at("x-target", default: "pdf")

<<<<<<< HEAD
#if(x-target != "md") {
  show: project.with(
    title: "Podstawy Sztucznej Inteligencji\nKamienie Milowe",
=======
#let temat = "Przewidywanie okresu wypowiedzenia umowy na podstawie danych klientów w celu zwiększenia zysku ze świadczenia usług"

#if(x-target != "md") {
  show: project.with(
    title: [Podstawy Sztucznej Inteligencji],
    subtitle: [#temat],
>>>>>>> ea2b078 (complete milestone 3)
    authors: (
      (name: "Bartosz Bodulski", affiliation: "Informatyka Techniczna"),
      (name: "Maciej Gładysiak", affiliation: "Informatyka Techniczna"),
    ),
<<<<<<< HEAD
    date: "April 22, 2026",
  )
outline()
=======
    date: "22 Kwietnia 2026",
  )
[
  #set text(lang: "pl")
  #outline()
]

>>>>>>> ea2b078 (complete milestone 3)
}

#pagebreak()

<<<<<<< HEAD
=======
#set text(lang: "pl")
>>>>>>> ea2b078 (complete milestone 3)
#include "stones/section1.typ"
#pagebreak() // nowa strona
#include "stones/section2.typ"
#pagebreak()
#include "stones/section3.typ"
