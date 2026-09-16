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

#  English is the source language: msgen copies every msgid into its msgstr,
#  so the catalogue never needs manual work and never falls behind.
mkdir -p locales/en/LC_MESSAGES
msgen locales/gutcheck.pot -o locales/en/LC_MESSAGES/gutcheck.po
#  msgen copies the .pot header verbatim, plural form placeholder included,
#  which Python's gettext refuses to parse. Fill it in.
sed -i 's/nplurals=INTEGER; plural=EXPRESSION;/nplurals=2; plural=(n != 1);/' \
    locales/en/LC_MESSAGES/gutcheck.po
sed -i '/^#, fuzzy$/d' locales/en/LC_MESSAGES/gutcheck.po

#  Every other language is merged, so existing translations survive.
for lang in fr; do
  po=locales/$lang/LC_MESSAGES/gutcheck.po
  if [ -f "$po" ]; then
    msgmerge --backup=off --update "$po" locales/gutcheck.pot
  else
    mkdir -p "$(dirname "$po")"
    msginit --no-translator --input=locales/gutcheck.pot --locale=$lang --output="$po"
  fi
done

for po in locales/*/LC_MESSAGES/gutcheck.po; do
  msgfmt "$po" -o "${po%.po}.mo"
done
