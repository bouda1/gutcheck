# gutcheck - internationalisation helpers
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

# i18n.py
import gettext
import locale
import os

lang_code = os.environ.get("GUTCHECK_LANG") or locale.getlocale()[0] or "en"
lang_code = lang_code.split(".")[0].split("_")[0]  # fr_FR.UTF-8 => fr

LANG_CODE = lang_code

LOCALE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")

lang = gettext.translation("gutcheck", localedir=LOCALE_DIR,
                           languages=[lang_code], fallback=True)
_ = lang.gettext
n_ = lang.ngettext


def N_(message):
    """Mark a string for extraction without translating it here.

    Used for literals stored in tables (food catalogues, ...) that are only
    translated later, through `_()`, once the actual value is known.
    """
    return message
