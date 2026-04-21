# Kompilacja `index.typ`
# Wymagania i instalacja

### typlite (konwersja do Markdown)
Linux
```bash
cargo install typlite
```
### typst (konwersja do PDF)
Linux, Windows
[https://typst.app/open-source/#download](https://typst.app/open-source/#download)

# Uruchomienie

## Linux
### Konwersja do PDF
```bash
typst compile index.typ
```
### Konwersja do Markdown
```bash
typlite index.typ README.md
```
#### Poprawa ścieżek
```bash
sed -i 's|\.\./images/|images/|g' README.md
sed -i 's|\.\./snippets/|snippets/|g' README.md
```
### Alternatywa
```bash
./compile.sh
```
Skrypt przeprowadza konwersje do PDF, do Markdown, poprawia ścieżki względne