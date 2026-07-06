"""M0 smoke test.

Proves the scaffolding works: the package is installed, importable via the
src/ layout, and exposes a version. Real logic tests arrive with M1's modules.
"""

import docforge


def test_package_imports_and_has_version() -> None:
    assert docforge.__version__ == "0.1.0"
