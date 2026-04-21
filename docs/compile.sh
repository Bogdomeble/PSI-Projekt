#!/bin/sh
typst compile index.typ
typlite index.typ README.md
sed -i 's|\.\./images/|images/|g' README.md
sed -i 's|\.\./snippets/|snippets/|g' README.md