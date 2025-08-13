"""
Unit tests for metadata tag access
"""

import pytest

from sasdata.metadata import Metadata, Process, access_meta, meta_tags


@pytest.mark.sasdata3
def test_tag_access():
    processes = [Process(name="Frobulator", date=None, description=None, terms={"Bobbin": "Threadbare"}, notes=[])]
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
    assert access_meta(meta, '.process[0].terms["Bobbin"]') == "Threadbare"


@pytest.mark.sasdata3
def test_tag_listing():
    processes = [Process(name="Frobulator", date=None, description=None, terms={"Bobbin": "Threadbare"}, notes=[])]
    meta = Metadata(
        title="Example",
        run=[5, 11],
        definition="An example metadata set",
        process=processes,
        sample=None,
        instrument=None,
        raw=None,
    )

    assert sorted(meta_tags(meta)) == [
        ".definition",
        ".instrument",
        ".process[0].date",
        ".process[0].description",
        ".process[0].name",
        '.process[0].terms["Bobbin"]',
        ".raw",
        ".run[0]",
        ".run[1]",
        ".sample",
        ".title",
    ]
