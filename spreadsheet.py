#!/usr/bin/env python3
# gutcheck - LibreOffice Calc (.ods) workbook reader
# Copyright (C) 2026 The gutcheck authors (see the AUTHORS file)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Reading LibreOffice Calc workbooks (.ods) without any external dependency.

An .ods file is a ZIP archive whose `content.xml` describes the sheets in the
OpenDocument format. We therefore read it directly with `zipfile` +
`xml.etree` rather than requiring odfpy or pandas.

Two points deserve attention:

  - Cells CARRY THEIR TYPE. A date entered in Calc becomes an
    `office:value-type="date"` cell whose displayed text follows the local
    format ("31/08/2026"), while the canonical value stays in
    `office:date-value` ("2026-08-31"). We always read the typed value when
    it exists, never the displayed text: otherwise the user's regional
    settings break the reading.

  - Empty cells and rows are COMPRESSED through `number-columns-repeated` /
    `number-rows-repeated` attributes, which commonly amount to 1024 or
    1048576 at the end of a sheet. They must be expanded, but with a bound,
    or a million empty rows get materialized.
"""

import re
import unicodedata
import zipfile
from xml.etree import ElementTree

from i18n import _

NS = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}
MAX_REPEATS = 4096          # bound on the expansion of empty cells
ISO_DURATION = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?")


def _text(cell):
    """Displayed text of a cell: the concatenation of its paragraphs."""
    parts = []
    for p in cell.findall("text:p", NS):
        parts.append("".join(p.itertext()))
    return " ".join(t.strip() for t in parts if t.strip())


def _value(cell):
    """Value of a cell, preferring the typed attribute over the display."""
    t = cell.get(f"{{{NS['office']}}}value-type")

    if t == "date":
        v = cell.get(f"{{{NS['office']}}}date-value", "")
        return v.split("T")[0] if v else _text(cell)

    if t == "time":
        v = cell.get(f"{{{NS['office']}}}time-value", "")
        m = ISO_DURATION.fullmatch(v) if v else None
        if m:
            h, mn = int(m.group(1) or 0), int(m.group(2) or 0)
            return f"{h:02d}:{mn:02d}"
        return _text(cell)

    if t in ("float", "percentage", "currency"):
        v = cell.get(f"{{{NS['office']}}}value", "")
        if v:
            f = float(v)
            return str(int(f)) if f == int(f) else str(f)
        return _text(cell)

    if t == "boolean":
        return cell.get(f"{{{NS['office']}}}boolean-value", "") or _text(cell)

    return _text(cell)


def _repeats(element, attribute, default=1):
    try:
        return max(1, min(int(element.get(attribute, default)), MAX_REPEATS))
    except (TypeError, ValueError):
        return default


def read_sheet(path, sheet=None):
    """Read one sheet of an .ods workbook.

    Args:
        path (str): path to the .ods file.
        sheet (str, optional): sheet name; defaults to the first sheet.

    Returns:
        list: list of rows, each row being a list of strings.
    """
    with zipfile.ZipFile(path) as z:
        root = ElementTree.fromstring(z.read("content.xml"))

    tables = root.findall(".//table:table", NS)
    if not tables:
        raise ValueError(_("no sheet in the workbook"))
    if sheet is None:
        table = tables[0]
    else:
        by_name = {t.get(f"{{{NS['table']}}}name"): t for t in tables}
        if sheet not in by_name:
            raise ValueError(_("sheet '{}' not found (available: {})").format(
                sheet, ", ".join(by_name)))
        table = by_name[sheet]

    rows = []
    for tr in table.findall(".//table:table-row", NS):
        cells = []
        for td in tr.findall("table:table-cell", NS):
            v = _value(td)
            cells.extend([v] * _repeats(
                td, f"{{{NS['table']}}}number-columns-repeated"))
        while cells and not cells[-1]:
            cells.pop()                          # trailing empty cells
        n = _repeats(tr, f"{{{NS['table']}}}number-rows-repeated")
        if not cells:
            n = 1                                # do not duplicate emptiness
        rows.extend([list(cells)] * n)

    while rows and not any(rows[-1]):
        rows.pop()
    return rows


def normalize_header(name):
    """ Normalize a header name by stripping whitespace, converting to lowercase, removing accents, and ignoring suffixes in parentheses. Spaces are replaced with underscores.

    Args:
        name (str): The header name to normalize.

    Returns:
        str: The normalized header name.
    """
    name = unicodedata.normalize("NFKD", name.strip().lower())
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.split("(")[0].strip().replace(" ", "_")


def read_ods(path, sheet=None):
    """
    Read an ODS file and return a list of dictionaries, similar to csv.DictReader (first line = headers).

    Args:
        path (str): Path to the ODS file.
        sheet (str, optional): Name of the sheet to read. Defaults to None (first sheet).

    Returns:
        list: A list of dictionaries representing the rows in the sheet, with normalized headers as keys.
    """
    lines = read_sheet(path, sheet)
    if not lines:
        return []
    headers = [normalize_header(c) for c in lines[0]]

    replacements = {
        normalize_header(_("date")): "date",
        normalize_header(_("time")): "time",
        normalize_header(_("foods")): "foods",
        normalize_header(_("meal")): "meal",
        normalize_header(_("pain")): "pain",
    }

    headers = [replacements.get(x, x) for x in headers]
    retval = []
    for line in lines[1:]:
        if not any(c.strip() for c in line):
            continue
        line = line + [""] * (len(headers) - len(line))
        retval.append(dict(zip(headers, line)))
    return retval


# ══════════════════════════════════════════════════════════════════
#  Writing
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

CONTENT_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
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
  <style:style style:name="{STYLE_TIME}" style:family="table-cell"
   style:data-style-name="Nh"/>
 </office:automatic-styles>
 <office:body><office:spreadsheet><table:table table:name="{sheet}">"""

CONTENT_FOOTER = """</table:table></office:spreadsheet></office:body>
</office:document-content>"""

#  A date/time cell carries its value in office:date-value, but its DISPLAY
#  comes from a data style referenced by table:style-name. Without that style,
#  Calc applies the "number" format and shows the internal value (days since
#  the epoch, fraction of a day for a time).
STYLE_DATE = "ceDate"
STYLE_TIME = "ceTime"

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
ISO_TIME = re.compile(r"(\d{1,2}):(\d{2})")


def _escape(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _cell(value):
    """Write a TYPED cell, the way Calc would: dates and times are not text,
    otherwise they get reformatted according to the local settings."""
    t = str(value).strip()
    if ISO_DATE.fullmatch(t):
        return (f'<table:table-cell table:style-name="{STYLE_DATE}" '
                f'office:value-type="date" office:date-value="{t}">'
                f'<text:p>{t}</text:p></table:table-cell>')
    m = ISO_TIME.fullmatch(t)
    if m:
        return (f'<table:table-cell table:style-name="{STYLE_TIME}" '
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
    return f'<table:table-cell office:value-type="string"><text:p>{_escape(t)}</text:p></table:table-cell>'


def write_ods(path, rows, sheet="diary"):
    """Write a minimal but valid .ods workbook, openable in Calc."""
    body = [CONTENT_HEADER.format(sheet=_escape(sheet),
                                  STYLE_DATE=STYLE_DATE,
                                  STYLE_TIME=STYLE_TIME)]
    for row in rows:
        body.append("<table:table-row>"
                    + "".join(_cell(c) for c in row)
                    + "</table:table-row>")
    body.append(CONTENT_FOOTER)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        # the mimetype must be the first entry, uncompressed
        z.writestr(zipfile.ZipInfo("mimetype"), MIMETYPE,
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/manifest.xml", MANIFEST.format(mime=MIMETYPE))
        z.writestr("styles.xml", STYLES)
        z.writestr("content.xml", "".join(body))
    return path
