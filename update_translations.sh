#!/bin/bash
xgettext -d gutcheck -o locales/gutcheck.pot --language=Python gutcheck.py simu.py spreadsheet.py test_gutcheck.py
for lang in fr en; do
  po=locales/$lang/LC_MESSAGES/gutcheck.po
  if [ -f "$po" ]; then
    msgmerge --update "$po" locales/gutcheck.pot   # met à jour sans écraser les traductions existantes
  else
    mkdir -p "$(dirname "$po")"
    msginit --input=locales/gutcheck.pot --locale=$lang --output="$po"
  fi
  msgfmt "$po" -o "${po%.po}.mo"
done
