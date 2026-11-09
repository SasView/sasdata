"""
Unit tests for the importer configuration, which says where in a file each item of metadata is found.
"""

import h5py
import numpy as np
import pytest
from lxml import etree

from sasdata.data_io.config import (
    METADATA_FIELDS,
    Hdf5Source,
    ImporterConfig,
    ImporterConfigError,
    TextSource,
    XmlSource,
)
from sasdata.data_io.importer import Importer
from sasdata.data_io.importers import AsciiImporter, HDF5Importer, SesansImporter, XmlImporter
from sasdata.metadata import Metadata
from sasdata.quantities import units
from test import local_data_finder, local_sesans_finder


def empty_metadata() -> Metadata:
    return Metadata.initialise_empty()


# ----------------------------------------------------------------------------------------------------------------------
# The configuration
# ----------------------------------------------------------------------------------------------------------------------

class TestConfiguration:
    def test_from_yaml(self):
        config = ImporterConfig.from_yaml("""
pathDict:
  title: /entry/title
  sample.thickness: sample/thickness
units:
  sample.thickness: mm
somethingElse: 3
""")
        assert config.pathDict == {"title": "/entry/title", "sample.thickness": "sample/thickness"}
        assert config.units == {"sample.thickness": "mm"}
        assert config.extra == {"somethingElse": 3}

    @pytest.mark.parametrize("text", ["", "   \n", "# nothing here"])
    def test_empty(self, text):
        config = ImporterConfig.from_yaml(text)
        assert config == ImporterConfig()
        assert not config

    def test_from_file(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("pathDict:\n  title: header:Title\n")
        assert ImporterConfig.from_file(path).pathDict == {"title": "header:Title"}

    def test_coerce(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("pathDict:\n  title: header:Title\n")
        expected = {"title": "header:Title"}
        assert ImporterConfig.coerce(None) == ImporterConfig()
        assert ImporterConfig.coerce({"pathDict": expected}).pathDict == expected
        assert ImporterConfig.coerce(path).pathDict == expected
        assert ImporterConfig.coerce(str(path)).pathDict == expected
        assert ImporterConfig.coerce("pathDict: {title: 'header:Title'}").pathDict == expected
        config = ImporterConfig(pathDict=expected)
        assert ImporterConfig.coerce(config) is config
        with pytest.raises(TypeError):
            ImporterConfig.coerce(3)

    @pytest.mark.parametrize("raw, message", [
        ({"pathDict": {"not.a.field": "x"}}, "Unknown metadata name"),
        ({"units": {"not.a.field": "mm"}}, "Unknown metadata name"),
        ({"units": {"sample.thickness": "not a unit"}}, "Cannot read the units"),
        ({"pathDict": ["title"]}, "must be a mapping"),
        ({"pathDict": {"title": " "}}, "is empty"),
    ])
    def test_invalid(self, raw, message):
        with pytest.raises(ImporterConfigError, match=message):
            ImporterConfig.from_dict(raw)

    def test_invalid_yaml(self):
        with pytest.raises(ImporterConfigError):
            ImporterConfig.from_yaml("pathDict: [")
        with pytest.raises(ImporterConfigError):
            ImporterConfig.from_yaml("- a list")

    def test_merged(self):
        base = ImporterConfig(pathDict={"title": "a", "definition": "b"}, units={"sample.thickness": "mm"})
        other = ImporterConfig(pathDict={"title": "c"}, units={"sample.thickness": "nm"})
        merged = base.merged(other)
        assert merged.pathDict == {"title": "c", "definition": "b"}
        assert merged.units == {"sample.thickness": "nm"}
        assert base.pathDict["title"] == "a"
        assert base.merged(None) is base

    def test_every_field_is_settable(self):
        """Each name that can be configured must be set on a metadata object with nothing in it"""
        for key, kind in METADATA_FIELDS.items():
            value = "1 mm" if kind == "quantity" else "1" if kind in ("float", "str") else "a"
            config = ImporterConfig(pathDict={key: "line:1"})
            metadata = config.apply_to_metadata(empty_metadata(), TextSource([value]))
            target = metadata
            for part in key.split("."):
                if part in ("detector", "process", "collimation"):
                    target = getattr(target, "collimations" if part == "collimation" else part)[0]
                else:
                    target = getattr(target, part)
            assert target is not None and not (isinstance(target, list) and not target), key


# ----------------------------------------------------------------------------------------------------------------------
# Finding values in each kind of file
# ----------------------------------------------------------------------------------------------------------------------

class TestTextSource:
    lines = ["# Title: My sample\n", "# Thickness = 1.5 mm\n", "Wavelength 6.0\n", "runs: run1 run2\n", "# run7\n"]

    def values(self, location):
        return [match.value for match in TextSource(self.lines).read(location)]

    def test_line(self):
        assert self.values("line:3") == ["Wavelength 6.0"]
        assert self.values("line:99") == []

    def test_header(self):
        assert self.values("header:Title") == ["My sample"]
        assert self.values("header:thickness") == ["1.5 mm"]
        assert self.values("header:Wavelength") == ["6.0"]
        assert self.values("header:Missing") == []

    def test_regex(self):
        assert self.values(r"regex:^runs: (\w+)") == ["run1"]
        assert self.values(r"regex:run\d") == ["run1", "run7"]

    @pytest.mark.parametrize("location", ["somewhere", "line:abc", "regex:("])
    def test_invalid_location(self, location):
        with pytest.raises(ImporterConfigError):
            TextSource(self.lines).read(location)

    def test_apply(self):
        config = ImporterConfig(pathDict={"title": "header:Title", "sample.thickness": "header:Thickness",
                                          "instrument.source.wavelength": "header:Wavelength"},
                                units={"instrument.source.wavelength": "nm"})
        metadata = config.apply_to_metadata(empty_metadata(), TextSource(self.lines))
        assert metadata.title == "My sample"
        assert metadata.sample.thickness.value == 1.5
        assert metadata.sample.thickness.in_units_of(units.meters) == pytest.approx(1.5e-3)
        assert metadata.instrument.source.wavelength.value == 6.0
        assert metadata.instrument.source.wavelength.in_units_of(units.meters) == pytest.approx(6e-9)

    def test_missing_location_is_skipped(self):
        config = ImporterConfig(pathDict={"title": "header:Missing"})
        assert config.apply_to_metadata(empty_metadata(), TextSource(self.lines)).title is None

    def test_units_are_required(self):
        config = ImporterConfig(pathDict={"sample.thickness": "line:3"})
        with pytest.raises(ImporterConfigError, match="units"):
            config.apply_to_metadata(empty_metadata(), TextSource(["5"]))

    def test_value_that_is_not_a_number(self):
        config = ImporterConfig(pathDict={"sample.transmission": "header:Title"})
        with pytest.raises(ImporterConfigError):
            config.apply_to_metadata(empty_metadata(), TextSource(self.lines))


class TestXmlSource:
    root = etree.fromstring(
        '<root xmlns="urn:test"><entry name="first"><Title>T</Title><sample><size unit="mm">2</size>'
        '<size>3</size></sample></entry></root>'
    )
    entry = root[0]

    def values(self, location):
        return [(match.value, match.attrs) for match in XmlSource(self.entry).read(location)]

    def test_relative_path_ignores_namespace(self):
        assert self.values("Title") == [("T", {})]
        assert self.values("sample/size") == [("2", {"unit": "mm"}), ("3", {})]

    def test_index_and_wildcard(self):
        assert self.values("sample/size[2]") == [("3", {})]
        assert self.values("sample/size[3]") == []
        assert self.values("*/size[1]") == [("2", {"unit": "mm"})]

    def test_absolute_path(self):
        assert self.values("/root/entry/Title") == [("T", {})]
        assert self.values("/other/entry/Title") == []

    def test_attribute(self):
        assert self.values("@name") == [("first", {})]
        assert self.values("sample/size@unit") == [("mm", {})]

    def test_missing(self):
        assert self.values("nothing") == []

    def test_quantity_takes_units_from_the_file(self):
        config = ImporterConfig(pathDict={"sample.thickness": "sample/size[1]"})
        metadata = config.apply_to_metadata(empty_metadata(), XmlSource(self.entry))
        assert metadata.sample.thickness.in_units_of(units.meters) == pytest.approx(2e-3)

    def test_configured_units_take_precedence(self):
        config = ImporterConfig(pathDict={"sample.thickness": "sample/size[1]"}, units={"sample.thickness": "nm"})
        metadata = config.apply_to_metadata(empty_metadata(), XmlSource(self.entry))
        assert metadata.sample.thickness.in_units_of(units.meters) == pytest.approx(2e-9)


class TestHdf5Source:
    @pytest.fixture
    def entry(self, tmp_path):
        with h5py.File(tmp_path / "source.h5", "w") as f:
            group = f.create_group("entry")
            group["title"] = np.array([b"A title"])
            group["sample/thickness"] = np.array([2.0])
            group["sample/thickness"].attrs["units"] = "mm"
            group["sample"].attrs["note"] = "a note"
            group["runs"] = np.array([b"one", b"two"])
            yield group

    def values(self, entry, location):
        return [(match.value, match.attrs) for match in Hdf5Source(entry).read(location)]

    def test_relative_and_absolute(self, entry):
        assert self.values(entry, "title") == [("A title", {})]
        assert self.values(entry, "/entry/title") == [("A title", {})]

    def test_attributes(self, entry):
        assert self.values(entry, "sample/thickness") == [(2.0, {"units": "mm"})]
        assert self.values(entry, "sample@note") == [("a note", {})]
        assert self.values(entry, "sample/thickness@units") == [("mm", {})]

    def test_several_values(self, entry):
        assert [value for value, _ in self.values(entry, "runs")] == ["one", "two"]

    def test_missing_and_groups(self, entry):
        assert self.values(entry, "missing") == []
        assert self.values(entry, "sample") == []

    def test_apply(self, entry):
        config = ImporterConfig(pathDict={"title": "title", "sample.thickness": "sample/thickness",
                                          "run": "runs", "sample.name": "sample@note"})
        metadata = config.apply_to_metadata(empty_metadata(), Hdf5Source(entry))
        assert metadata.title == "A title"
        assert metadata.run == ["one", "two"]
        assert metadata.sample.name == "a note"
        assert metadata.sample.thickness.in_units_of(units.meters) == pytest.approx(2e-3)


# ----------------------------------------------------------------------------------------------------------------------
# The importers
# ----------------------------------------------------------------------------------------------------------------------

class TestImporters:
    def test_importer_keeps_its_configuration(self):
        assert Importer({"pathDict": {"title": "line:1"}}).config.pathDict == {"title": "line:1"}
        assert Importer().config == ImporterConfig()
        assert XmlImporter("pathDict: {title: Title}").config.pathDict == {"title": "Title"}

    def test_xml(self):
        path = local_data_finder("cansas1d.xml")
        original = next(iter(Importer.load(path).values()))
        config = {"pathDict": {"sample.name": "Title", "sample.sample_id": "/SASroot/SASentry/Run"}}
        configured = next(iter(Importer.load(path, config).values()))
        assert configured.metadata.sample.name == original.metadata.title
        assert configured.metadata.sample.sample_id == original.metadata.run[0]
        # Metadata that is not configured is unchanged
        assert configured.metadata.title == original.metadata.title
        assert configured.metadata.instrument == original.metadata.instrument

    def test_hdf5(self):
        path = local_data_finder("simpleexamplefile.h5")
        original = next(iter(HDF5Importer().load_data(path).values()))
        if original.metadata.title is None:
            pytest.skip("The file has no title")
        configured = next(iter(HDF5Importer({"pathDict": {"definition": "title"}}).load_data(path).values()))
        assert configured.metadata.definition == original.metadata.title

    def test_sesans(self):
        path = local_sesans_finder("sphere2micron.ses")
        configured = SesansImporter({"pathDict": {"title": "line:1"}}).load_data(path)
        assert next(iter(configured.values())).metadata.title.startswith("FileFormatVersion")

    def test_ascii(self, tmp_path):
        path = tmp_path / "data.txt"
        rows = "\n".join(f"{0.01 * i} {10 / i} {0.1 / i}" for i in range(1, 8))
        path.write_text(f"# Sample: polystyrene\n# Thickness: 1.5 mm\n{rows}\n")
        config = {"pathDict": {"sample.name": "header:Sample", "sample.thickness": "header:Thickness"}}
        measurement = next(iter(AsciiImporter(config).load_data(path).values()))
        assert measurement.metadata.sample.name == "polystyrene"
        assert measurement.metadata.sample.thickness.in_units_of(units.meters) == pytest.approx(1.5e-3)

    def test_per_file_configuration(self, tmp_path):
        paths = []
        for name in ("a", "b"):
            path = tmp_path / f"{name}.txt"
            rows = "\n".join(f"{0.01 * i} {10 / i} {0.1 / i}" for i in range(1, 8))
            path.write_text(f"# Sample: {name}\n{rows}\n")
            paths.append(path)
        configs = [{"pathDict": {"sample.name": "header:Sample"}}, {"pathDict": {"sample.name": "line:1"}}]
        measurements, errors = Importer().import_data(paths, configs)
        assert errors == []
        assert [m.metadata.sample.name for m in measurements] == ["a", "# Sample: b"]

    def test_invalid_configuration_is_reported(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("\n".join(f"{0.01 * i} {10 / i} {0.1 / i}" for i in range(1, 8)) + "\n")
        _, errors = Importer().import_data([path], [{"pathDict": {"sample.name": "nonsense"}}])
        assert len(errors) == 1 and "not a location in a text file" in errors[0]
