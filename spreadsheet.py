#!/usr/bin/env python3
"""
Lecture de classeurs LibreOffice Calc (.ods) sans dépendance externe.

Un .ods est une archive ZIP dont `content.xml` décrit les feuilles au format
OpenDocument. On lit donc directement avec `zipfile` + `xml.etree` plutôt que
d'imposer odfpy ou pandas.

Deux points méritent l'attention :

  - Les cellules PORTENT LEUR TYPE. Une date saisie dans Calc devient une
    cellule `office:value-type="date"` dont le texte affiché suit le format
    local (« 31/08/2026 »), pendant que la valeur canonique reste dans
    `office:date-value` (« 2026-08-31 »). On lit toujours la valeur typée
    quand elle existe, jamais le texte affiché : sinon le format régional de
    l'utilisateur casse la lecture.

  - Les cellules et lignes vides sont COMPRESSÉES par des attributs
    `number-columns-repeated` / `number-rows-repeated`, qui valent couramment
    1024 ou 1048576 en fin de feuille. Il faut les développer, mais avec une
    borne, sous peine de matérialiser un million de lignes vides.
"""

import re
import unicodedata
import zipfile
from xml.etree import ElementTree

NS = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}
MAX_REPETITIONS = 4096          # borne sur le développement des cellules vides
DUREE_ISO = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?")


def _texte(cellule):
    """Texte affiché d'une cellule : concaténation de ses paragraphes."""
    parts = []
    for p in cellule.findall("text:p", NS):
        parts.append("".join(p.itertext()))
    return " ".join(t.strip() for t in parts if t.strip())


def _valeur(cellule):
    """Valeur d'une cellule, en privilégiant l'attribut typé sur l'affichage."""
    t = cellule.get(f"{{{NS['office']}}}value-type")

    if t == "date":
        v = cellule.get(f"{{{NS['office']}}}date-value", "")
        return v.split("T")[0] if v else _texte(cellule)

    if t == "time":
        v = cellule.get(f"{{{NS['office']}}}time-value", "")
        m = DUREE_ISO.fullmatch(v) if v else None
        if m:
            h, mn = int(m.group(1) or 0), int(m.group(2) or 0)
            return f"{h:02d}:{mn:02d}"
        return _texte(cellule)

    if t in ("float", "percentage", "currency"):
        v = cellule.get(f"{{{NS['office']}}}value", "")
        if v:
            f = float(v)
            return str(int(f)) if f == int(f) else str(f)
        return _texte(cellule)

    if t == "boolean":
        return cellule.get(f"{{{NS['office']}}}boolean-value", "") or _texte(cellule)

    return _texte(cellule)


def _repetitions(element, attribut, defaut=1):
    try:
        return max(1, min(int(element.get(attribut, defaut)), MAX_REPETITIONS))
    except (TypeError, ValueError):
        return defaut


def lire_feuille(chemin, feuille=None):
    """→ liste de lignes, chaque ligne étant une liste de chaînes."""
    with zipfile.ZipFile(chemin) as z:
        racine = ElementTree.fromstring(z.read("content.xml"))

    tables = racine.findall(".//table:table", NS)
    if not tables:
        raise ValueError("aucune feuille dans le classeur")
    if feuille is None:
        table = tables[0]
    else:
        par_nom = {t.get(f"{{{NS['table']}}}name"): t for t in tables}
        if feuille not in par_nom:
            raise ValueError(f"feuille « {feuille} » absente "
                             f"(disponibles : {', '.join(par_nom)})")
        table = par_nom[feuille]

    lignes = []
    for tr in table.findall(".//table:table-row", NS):
        cellules = []
        for td in tr.findall("table:table-cell", NS):
            v = _valeur(td)
            cellules.extend([v] * _repetitions(
                td, f"{{{NS['table']}}}number-columns-repeated"))
        while cellules and not cellules[-1]:
            cellules.pop()                       # queue de cellules vides
        n = _repetitions(tr, f"{{{NS['table']}}}number-rows-repeated")
        if not cellules:
            n = 1                                # ne pas dupliquer le vide
        lignes.extend([list(cellules)] * n)

    while lignes and not any(lignes[-1]):
        lignes.pop()
    return lignes


def normaliser_entete(nom):
    """« Douleur (0-10) » → « douleur » : casse, accents et suffixes ignorés."""
    nom = unicodedata.normalize("NFKD", nom.strip().lower())
    nom = "".join(c for c in nom if not unicodedata.combining(c))
    return nom.split("(")[0].strip().replace(" ", "_")


def read_ods(chemin, feuille=None):
    """
    Read an ODS file and return a list of dictionaries, similar to csv.DictReader (first line = headers).

    Args:
        chemin (str): Path to the ODS file.
        feuille (str, optional): Name of the sheet to read. Defaults to None (first sheet).

    Returns:
        list: A list of dictionaries representing the rows in the sheet, with normalized headers as keys.
    """
    lignes = lire_feuille(chemin, feuille)
    if not lignes:
        return []
    entetes = [normaliser_entete(c) for c in lignes[0]]
    sortie = []
    for ligne in lignes[1:]:
        if not any(c.strip() for c in ligne):
            continue
        ligne = ligne + [""] * (len(entetes) - len(ligne))
        sortie.append(dict(zip(entetes, ligne)))
    return sortie


# ══════════════════════════════════════════════════════════════════
#  Écriture
# ══════════════════════════════════════════════════════════════════

MIMETYPE = "application/vnd.oasis.opendocument.spreadsheet"

MANIFEST = """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">
 <manifest:file-entry manifest:full-path="/" manifest:media-type="{mime}"/>
 <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
 <manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"""

STYLES = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-styles
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 office:version="1.2">
 <office:styles>
  <style:style style:name="Default" style:family="table-cell"/>
 </office:styles>
</office:document-styles>"""

ENTETE_CONTENU = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:number="urn:oasis:names:tc:opendocument:xmlns:datastyle:1.0"
 office:version="1.2">
 <office:automatic-styles>
  <number:date-style style:name="Nd">
   <number:year number:style="long"/><number:text>-</number:text>
   <number:month number:style="long"/><number:text>-</number:text>
   <number:day number:style="long"/>
  </number:date-style>
  <number:time-style style:name="Nh">
   <number:hours number:style="long"/><number:text>:</number:text>
   <number:minutes number:style="long"/>
  </number:time-style>
  <style:style style:name="{STYLE_DATE}" style:family="table-cell"
   style:data-style-name="Nd"/>
  <style:style style:name="{STYLE_HEURE}" style:family="table-cell"
   style:data-style-name="Nh"/>
 </office:automatic-styles>
 <office:body><office:spreadsheet><table:table table:name="{feuille}">"""

PIED_CONTENU = """</table:table></office:spreadsheet></office:body>
</office:document-content>"""

#  Une cellule date/heure porte sa valeur dans office:date-value, mais son
#  AFFICHAGE vient d'un style de données référencé par table:style-name. Sans
#  ce style, Calc applique le format « nombre » et montre la valeur interne
#  (nombre de jours depuis l'époque, fraction de journée pour une heure).
STYLE_DATE = "ceDate"
STYLE_HEURE = "ceHeure"

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
ISO_HEURE = re.compile(r"(\d{1,2}):(\d{2})")


def _echapper(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _cellule(valeur):
    """Écrit une cellule TYPÉE, comme le ferait Calc : dates et heures ne sont
    pas du texte, sinon elles se reformatent au gré des paramètres locaux."""
    t = str(valeur).strip()
    if ISO_DATE.fullmatch(t):
        return (f'<table:table-cell table:style-name="{STYLE_DATE}" '
                f'office:value-type="date" office:date-value="{t}">'
                f'<text:p>{t}</text:p></table:table-cell>')
    m = ISO_HEURE.fullmatch(t)
    if m:
        return (f'<table:table-cell table:style-name="{STYLE_HEURE}" '
                f'office:value-type="time" '
                f'office:time-value="PT{int(m.group(1)):02d}H{m.group(2)}M00S">'
                f'<text:p>{t}</text:p></table:table-cell>')
    try:
        f = float(t)
        return (f'<table:table-cell office:value-type="float" office:value="{f}">'
                f'<text:p>{t}</text:p></table:table-cell>')
    except ValueError:
        pass
    if not t:
        return '<table:table-cell/>'
    return f'<table:table-cell office:value-type="string"><text:p>{_echapper(t)}</text:p></table:table-cell>'


def ecrire_ods(chemin, lignes, feuille="journal"):
    """Écrit un classeur .ods minimal mais valide, ouvrable dans Calc."""
    corps = [ENTETE_CONTENU.format(feuille=_echapper(feuille),
                                   STYLE_DATE=STYLE_DATE,
                                   STYLE_HEURE=STYLE_HEURE)]
    for ligne in lignes:
        corps.append("<table:table-row>"
                     + "".join(_cellule(c) for c in ligne)
                     + "</table:table-row>")
    corps.append(PIED_CONTENU)

    with zipfile.ZipFile(chemin, "w", zipfile.ZIP_DEFLATED) as z:
        # le mimetype doit être la première entrée, non compressée
        z.writestr(zipfile.ZipInfo("mimetype"), MIMETYPE,
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/manifest.xml", MANIFEST.format(mime=MIMETYPE))
        z.writestr("styles.xml", STYLES)
        z.writestr("content.xml", "".join(corps))
    return chemin
