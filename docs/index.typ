// Kamienie milowe w oddzielnych plikach później
// w trakcie pisania nowego można tutaj i potem ctrl+x ctrl+v
#import "template.typ": *
#show heading: smallcaps
#show link: set text(blue)
#show link: emph
#set text(lang: "pl")

#let x-target = sys.inputs.at("x-target", default: "pdf")

#if(x-target != "md") {
  show: project.with(
    title: "Podstawy Sztucznej Inteligencji\nKamienie Milowe",
    authors: (
      (name: "Bartosz Bodulski", affiliation: "Informatyka Techniczna"),
      (name: "Maciej Gładysiak", affiliation: "Informatyka Techniczna"),
    ),
    date: "April 22, 2026",
  )
outline()
}

#pagebreak()

#include "stones/section1.typ"
#pagebreak() // nowa strona
#include "stones/section2.typ"
#pagebreak()
#include "stones/section3.typ"
