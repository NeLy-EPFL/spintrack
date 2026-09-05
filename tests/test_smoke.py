import re

import spintrack


def test_core_module_loads_and_reports_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", spintrack.__version__)
