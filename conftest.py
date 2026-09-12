# gutcheck - pytest configuration
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

"""Pin the language of the test run.

The test suite asserts on user-facing messages, which are translated at
import time by i18n.py. Forcing the source language (English) keeps the
suite independent from the developer's locale. This runs before any test
module — and therefore before i18n — is imported.
"""

import os

os.environ.setdefault("GUTCHECK_LANG", "en")


def pytest_configure(config):
    """Declare the markers used by the suite."""
    config.addinivalue_line(
        "markers",
        "slow: end-to-end check on a synthetic diary; a few seconds to run",
    )
