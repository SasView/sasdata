"""
Unit tests for metadata tag access
"""

import pytest

from sasdata.metadata import Metadata, Process, access_meta


@pytest.mark.sasdata3
def test_tag_access():
    processes = [Process(name="Frobulator", date=None, description=None, terms={}, notes=[])]
    meta = Metadata(
        title="Example",
        run=[5, 11],
        definition="An example metadata set",
        process=processes,
        sample=None,
        instrument=None,
        raw=None,
    )

    assert access_meta(meta, ".title") == "Example"
    assert access_meta(meta, ".run") == [5, 11]
    assert access_meta(meta, ".process")[0].name == "Frobulator"
    assert access_meta(meta, ".process[0].name") == "Frobulator"
