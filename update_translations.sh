#!/bin/bash
#  Rebuild locales/gutcheck.pot from the source strings (English), merge it
#  into every .po catalogue without losing existing translations, then
#  compile the .mo files actually loaded at runtime.
set -e

SOURCES="gutcheck.py model.py simu.py spreadsheet.py test_gutcheck.py"

xgettext -d gutcheck -o locales/gutcheck.pot --language=Python \
         --from-code=UTF-8 --keyword=_ --keyword=N_ --keyword=n_:1,2 \
         --package-name=gutcheck --copyright-holder="The gutcheck authors" \
         $SOURCES

for lang in fr en; do
  po=locales/$lang/LC_MESSAGES/gutcheck.po
  if [ -f "$po" ]; then
    msgmerge --backup=off --update "$po" locales/gutcheck.pot
  else
    mkdir -p "$(dirname "$po")"
    msginit --no-translator --input=locales/gutcheck.pot --locale=$lang --output="$po"
  fi
  msgfmt "$po" -o "${po%.po}.mo"
done
