"""
Unit tests for metadata tag access
"""

import pytest

from sasdata.metadata import Metadata, Process, access_meta, collect_tags, meta_tags


@pytest.mark.sasdata
def test_tag_access():
    """Check ability to access terms by name"""
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


@pytest.mark.sasdata
def test_tag_listing():
    """Check the ability to collect term names from object"""
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


@pytest.mark.sasdata
def test_tag_merge():
    """Check ability to access terms by name"""
    processes = [
        Process(
            name="Frobulator",
            date=None,
            description=None,
            terms={"Bobbin": "Threadbare", "Guybrush": "Threepwood"},
            notes=[],
        )
    ]
    meta = Metadata(
        title="Example",
        run=[5, 11],
        definition="An example metadata set",
        process=processes,
        sample=None,
        instrument=None,
        raw=None,
    )

    processes2 = [
        Process(
            name="Frobulator",
            date=None,
            description=None,
            terms={"Bobbin": "Threadbare", "Doctor Ed": "Edison"},
            notes=[],
        )
    ]
    meta2 = Metadata(
        title="Example 2",
        run=[5, 11, 12],
        definition="A second example metadata set",
        process=processes2,
        sample=None,
        instrument=None,
        raw=None,
    )

    result = collect_tags([meta, meta2])

    assert result.singular == set(
        [
            ".process[0].name",
            ".process[0].date",
            ".process[0].description",
            '.process[0].terms["Bobbin"]',
            ".run[0]",
            ".run[1]",
            ".sample",
            ".instrument",
            ".raw",
        ]
    )

    # Only meta2 has a run[2] term, so it cannot be collected
    assert ".run[2]" not in result.singular

    assert result.variable == set([".definition", ".title"])
