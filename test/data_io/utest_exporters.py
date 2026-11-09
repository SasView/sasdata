"""
Unit tests for the Exporter class and the NXcanSAS, canSAS XML, ASCII and CSV exporters.

Every exporter is tested by exporting data loaded from the test files and importing it again, with both the sasdata
importers and the data loader used by SasView, to check that exported data can be read back in.
"""

import os
import stat
from pathlib import Path

import h5py
import numpy as np
import pytest
from lxml import etree

from sasdata.data import SasMeasurement
from sasdata.data_io.exporter import Exporter, ExportError, default_file_mode
from sasdata.data_io.exporters import AsciiExporter, CanSASXmlExporter, CsvExporter, NXcanSASExporter
from sasdata.data_io.importer import Importer
from sasdata.dataloader.data_info import Data1D, Data2D
from sasdata.dataloader.loader import Loader
from sasdata.dataset_types import one_dim, sesans, two_dim
from sasdata.quantities import units
from sasdata.quantities.quantity import NamedQuantity
from sasdata.quantities.units import Unit
from test import local_data_finder, local_sesans_finder

SCHEMA = Path(__file__).parents[2] / "sasdata" / "dataloader" / "readers" / "schema" / "cansas1d_v1_1.xsd"

# Source files for the round trip tests, and the formats each one can be exported to.
ONE_DIM_FILES = ["ISIS_1_1.xml", "cansas1d.xml", "cansas1d_slit.xml", "simpleexamplefile.h5", "test_3_columns.txt"]
TWO_DIM_FILES = ["MAR07232_rest.h5"]
MIXED_FILES = ["nxcansas_1Dand2D_multisasentry.h5"]

# The format each exporter writes, as identified by file_format
EXPECTED_FORMAT = {
    NXcanSASExporter: "hdf5",
    CanSASXmlExporter: "xml",
    AsciiExporter: "ascii",
    CsvExporter: "ascii",
}


def file_format(path: str | Path) -> str:
    """Identify the format of a file from its contents, so the extension of an exported file is not relied on"""
    if h5py.is_hdf5(path):
        return "hdf5"
    with open(path, "rb") as f:
        if f.read(1024).lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"<"):
            return "xml"
    return "ascii"


def import_file(path: str | Path) -> dict[str, SasMeasurement]:
    """Import a file, letting the Importer infer the format from the file contents rather than its extension"""
    return Importer.load(path)


def load(filename: str) -> dict[str, SasMeasurement]:
    return import_file(local_data_finder(filename))


def first(data: dict[str, SasMeasurement]) -> SasMeasurement:
    return next(iter(data.values()))


def make_1d(name: str = "synthetic", points: int = 20, slit: bool = False,
            q_units: Unit = units.per_nanometer) -> SasMeasurement:
    """
    Synthetic 1D data, with Q in nm^-1 by default so the exporters are tested with units other than those SasView
    assumes. Use ``q_units=units.per_angstrom`` for data that SasView's loaders can read correctly.
    """
    q = np.linspace(0.001, 0.2, points)
    q_scale = 1.0 if q_units == units.per_angstrom else 10.0
    contents = {
        "Q": NamedQuantity("Q", q * q_scale, q_units, standard_error=q * 0.5),
        "I": NamedQuantity("I", 1 / q, units.per_centimeter, standard_error=0.1 / q),
    }
    if slit:
        contents["Q"] = NamedQuantity("Q", q, units.per_angstrom)
        contents["dQw"] = NamedQuantity("dQw", np.full(points, 0.001), units.per_angstrom)
        contents["dQl"] = NamedQuantity("dQl", np.full(points, 0.1), units.per_angstrom)
    return SasMeasurement(name, contents, one_dim)


def make_2d(name: str = "synthetic 2D", nx: int = 12, ny: int = 10) -> SasMeasurement:
    qx, qy = np.meshgrid(np.linspace(-0.1, 0.1, nx), np.linspace(-0.08, 0.08, ny))
    intensity = np.exp(-(qx ** 2 + qy ** 2) * 100)
    contents = {
        "Qx": NamedQuantity("Qx", qx.reshape(-1), units.per_angstrom),
        "Qy": NamedQuantity("Qy", qy.reshape(-1), units.per_angstrom),
        "I": NamedQuantity("I", intensity.reshape(-1), units.per_centimeter, standard_error=intensity.reshape(-1) / 10),
    }
    return SasMeasurement(name, contents, two_dim)


def assert_same_data(original: SasMeasurement, imported: SasMeasurement):
    """
    The quantities (and their uncertainties) must match, allowing for changes of array shape. The exporters keep the
    units of the data as they are, so the units must not change either.
    """
    assert imported.dataset_type.name == original.dataset_type.name
    imported_contents = imported._data_contents
    for name, quantity in original._data_contents.items():
        if name == "dQ" and name not in imported_contents:
            # Formats which only hold one Q resolution store dQ as the uncertainty in Q
            assert np.allclose(np.ravel(quantity.in_units_of(imported_contents["Q"].units)),
                               np.ravel(imported_contents["Q"].standard_error.value), equal_nan=True)
            continue
        assert name in imported_contents, f"{name} was not imported"
        result = imported_contents[name]
        assert result.units == quantity.units, f"{name} changed from {quantity.units} to {result.units}"
        assert np.allclose(np.ravel(quantity.in_units_of(result.units)), np.ravel(result.value), equal_nan=True)
        if quantity.has_error:
            assert result.has_error, f"The uncertainty in {name} was not imported"
            assert np.allclose(np.ravel(quantity.standard_error.in_units_of(result.units)),
                               np.ravel(result.standard_error.value), equal_nan=True)


def legacy_load(path: Path) -> list[Data1D | Data2D]:
    """Load a file with the data loader used by SasView"""
    loaded = Loader().load(str(path))
    for datum in loaded:
        assert not datum.errors, f"SasView reported errors loading {path}: {datum.errors}"
    return loaded


def assert_legacy_matches(original: SasMeasurement, legacy: Data1D | Data2D):
    """
    The data read by SasView must match the original data.

    SasView assumes 1/Å and 1/cm for text files and does not convert the units of HDF5 files, and the exporters keep the
    units of the data as they are, so the original data must already be in SasView's units to be read back correctly.
    """
    contents = original._data_contents
    for name, unit in {"Q": units.per_angstrom, "Qx": units.per_angstrom, "Qy": units.per_angstrom,
                       "I": units.per_centimeter}.items():
        if name in contents:
            assert contents[name].units == unit, f"{name} is not in SasView's units, so SasView cannot read it back"
    if original.dataset_type == one_dim:
        assert isinstance(legacy, Data1D)
        q = contents["Q"].in_units_of(units.per_angstrom)
        # Some of SasView's readers drop points at Q = 0, and all of them sort the data by Q
        kept = np.arange(q.size) if legacy.x.size == q.size else np.flatnonzero(q != 0)
        order = kept[np.argsort(q[kept], kind="stable")]
        legacy_order = np.argsort(legacy.x, kind="stable")
        assert np.allclose(legacy.x[legacy_order], q[order])
        assert np.allclose(legacy.y[legacy_order], contents["I"].in_units_of(units.per_centimeter)[order],
                           equal_nan=True)
        if contents["I"].has_error:
            assert np.allclose(legacy.dy[legacy_order],
                               contents["I"].standard_error.in_units_of(units.per_centimeter)[order], equal_nan=True)
    else:
        assert isinstance(legacy, Data2D)
        assert np.allclose(np.ravel(legacy.qx_data), np.ravel(contents["Qx"].in_units_of(units.per_angstrom)))
        assert np.allclose(np.ravel(legacy.qy_data), np.ravel(contents["Qy"].in_units_of(units.per_angstrom)))
        assert np.allclose(np.ravel(legacy.data), np.ravel(contents["I"].in_units_of(units.per_centimeter)),
                           equal_nan=True)


# ----------------------------------------------------------------------------------------------------------------------
# Exporter selection
# ----------------------------------------------------------------------------------------------------------------------

class TestExporterSelection:
    def test_registry(self):
        exporters = Exporter.exporters()
        for exporter in (NXcanSASExporter, CanSASXmlExporter, AsciiExporter, CsvExporter):
            assert exporter in exporters
            assert issubclass(exporter, Exporter)
        assert len(exporters) == 4

    @pytest.mark.parametrize("filename, expected", [
        ("data.h5", NXcanSASExporter),
        ("data.H5", NXcanSASExporter),
        ("data.nxs", NXcanSASExporter),
        ("data.hdf5", NXcanSASExporter),
        ("data.xml", CanSASXmlExporter),
        ("data.txt", AsciiExporter),
        ("data.dat", AsciiExporter),
        ("data.csv", CsvExporter),
        ("data", None),
        ("data.unknown", None),
    ])
    def test_find_exporter_by_extension(self, filename, expected):
        assert Exporter.find_exporter(path=filename) is expected

    @pytest.mark.parametrize("name, expected", [
        ("NXcanSAS", NXcanSASExporter),
        ("nxcansas", NXcanSASExporter),
        ("HDF5", NXcanSASExporter),
        ("canSAS XML", CanSASXmlExporter),
        ("xml", CanSASXmlExporter),
        ("ASCII", AsciiExporter),
        ("csv", CsvExporter),
        (CsvExporter, CsvExporter),
        ("not a format", None),
    ])
    def test_find_exporter_by_name(self, name, expected):
        assert Exporter.find_exporter(name=name) is expected

    def test_name_overrides_extension(self, tmp_path):
        """An exporter chosen by name is used whatever the file extension, and the importer finds it by content"""
        data = load("ISIS_1_1.xml")
        destination = tmp_path / "exported.dat"
        assert Exporter().export_data({destination: data}, exporter="canSAS XML") == []
        assert file_format(destination) == "xml"
        assert_same_data(first(data), first(import_file(destination)))

    def test_child_exporter_ignores_extension(self, tmp_path):
        destination = tmp_path / "no_extension"
        assert NXcanSASExporter().export_data({destination: make_1d()}) == []
        assert file_format(destination) == "hdf5"

    def test_unknown_extension(self, tmp_path):
        errors = Exporter().export_data({tmp_path / "data.unknown": make_1d()})
        assert len(errors) == 1
        assert "Unable to determine the export format" in errors[0]
        assert not (tmp_path / "data.unknown").exists()

    def test_unknown_exporter(self, tmp_path):
        errors = Exporter().export_data({tmp_path / "data.h5": make_1d()}, exporter="not a format")
        assert len(errors) == 1
        assert "Unknown exporter" in errors[0]


# ----------------------------------------------------------------------------------------------------------------------
# Destination validation
# ----------------------------------------------------------------------------------------------------------------------

class TestDestinationValidation:
    def test_valid_destination(self, tmp_path):
        assert Exporter.validate_destination(tmp_path / "data.h5") == (tmp_path / "data.h5").absolute()
        # The check must not leave anything behind
        assert list(tmp_path.iterdir()) == []

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Exporter.validate_destination(tmp_path / "missing" / "data.h5")
        errors = Exporter().export_data({tmp_path / "missing" / "data.h5": make_1d()})
        assert len(errors) == 1 and "does not exist" in errors[0]

    def test_parent_is_a_file(self, tmp_path):
        (tmp_path / "file").write_text("")
        with pytest.raises(NotADirectoryError):
            Exporter.validate_destination(tmp_path / "file" / "data.h5")

    def test_destination_is_a_directory(self, tmp_path):
        (tmp_path / "data.h5").mkdir()
        with pytest.raises(IsADirectoryError):
            Exporter.validate_destination(tmp_path / "data.h5")

    def test_no_overwrite(self, tmp_path):
        destination = tmp_path / "data.csv"
        destination.write_text("original")
        with pytest.raises(FileExistsError):
            Exporter.validate_destination(destination, overwrite=False)
        errors = Exporter().export_data({destination: make_1d()}, overwrite=False)
        assert len(errors) == 1
        assert destination.read_text() == "original"

    def test_overwrite(self, tmp_path):
        destination = tmp_path / "data.csv"
        destination.write_text("original")
        assert Exporter().export_data({destination: make_1d()}) == []
        assert_same_data(make_1d(), first(import_file(destination)))

    def test_directory_not_writable(self, tmp_path, monkeypatch):
        """os.access is patched so the test also works when run with administrator rights"""
        real_access = os.access
        monkeypatch.setattr(os, "access", lambda p, mode: False if Path(p) == tmp_path else real_access(p, mode))
        with pytest.raises(PermissionError):
            Exporter.validate_destination(tmp_path / "data.h5")
        errors = Exporter().export_data({tmp_path / "data.h5": make_1d()})
        assert len(errors) == 1 and "permission" in errors[0]
        assert not (tmp_path / "data.h5").exists()

    def test_file_not_writable(self, tmp_path, monkeypatch):
        destination = tmp_path / "data.h5"
        destination.write_text("original")
        real_access = os.access
        monkeypatch.setattr(os, "access", lambda p, mode: False if Path(p) == destination else real_access(p, mode))
        with pytest.raises(PermissionError):
            Exporter.validate_destination(destination)

    def test_directory_cannot_be_written(self, tmp_path, monkeypatch):
        """Permission systems that os.access does not understand are caught by trying to create a file"""
        def refuse(*args, **kwargs):
            raise PermissionError("Access is denied")
        monkeypatch.setattr("tempfile.mkstemp", refuse)
        with pytest.raises(PermissionError):
            Exporter.validate_destination(tmp_path / "data.h5")

    def test_failed_write_keeps_existing_file(self, tmp_path, monkeypatch):
        destination = tmp_path / "data.xml"
        destination.write_text("original")

        def fail(self, data, path):
            Path(path).write_text("partial")
            raise RuntimeError("Write failed")
        monkeypatch.setattr(CanSASXmlExporter, "write", fail)
        errors = Exporter().export_data({destination: make_1d()})
        assert len(errors) == 1 and "Write failed" in errors[0]
        assert destination.read_text() == "original"
        assert [p.name for p in tmp_path.iterdir()] == ["data.xml"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
    def test_exported_file_permissions(self, tmp_path):
        destination = tmp_path / "data.h5"
        Exporter.export(make_1d(), destination)
        # The export is written via a private temporary file, but must end up with normal file permissions
        assert stat.S_IMODE(destination.stat().st_mode) == default_file_mode()


# ----------------------------------------------------------------------------------------------------------------------
# Data validation
# ----------------------------------------------------------------------------------------------------------------------

class TestDataValidation:
    @pytest.mark.parametrize("exporter, extension", [(CanSASXmlExporter, "xml"), (CsvExporter, "csv")])
    def test_one_dim_only(self, tmp_path, exporter, extension):
        errors = Exporter().export_data({tmp_path / f"data.{extension}": make_2d()})
        assert len(errors) == 1 and "only supports" in errors[0]
        with pytest.raises(ExportError):
            exporter.export(make_2d(), tmp_path / f"data.{extension}")

    @pytest.mark.parametrize("extension", ["h5", "xml", "txt", "csv"])
    def test_sesans_not_supported(self, tmp_path, extension):
        data = first(Importer.load(local_sesans_finder("sphere2micron.ses")))
        assert data.dataset_type == sesans
        errors = Exporter().export_data({tmp_path / f"data.{extension}": data})
        assert len(errors) == 1 and "only supports" in errors[0]

    @pytest.mark.parametrize("extension", ["txt", "csv"])
    def test_one_dataset_per_text_file(self, tmp_path, extension):
        errors = Exporter().export_data({tmp_path / f"data.{extension}": [make_1d("a"), make_1d("b")]})
        assert len(errors) == 1 and "one dataset per file" in errors[0]

    def test_no_data(self, tmp_path):
        errors = Exporter().export_data({tmp_path / "data.h5": []})
        assert len(errors) == 1 and "no data" in errors[0]

    def test_errors_for_each_file(self, tmp_path):
        """A failure exporting one file does not stop the others being exported"""
        errors = Exporter().export_data({
            tmp_path / "good.h5": make_1d(),
            tmp_path / "bad.xml": make_2d(),
            tmp_path / "missing" / "bad.csv": make_1d(),
            tmp_path / "good.txt": make_1d(),
        })
        assert len(errors) == 2
        assert sorted(p.name for p in tmp_path.iterdir()) == ["good.h5", "good.txt"]


# ----------------------------------------------------------------------------------------------------------------------
# Round trips: export, then import with sasdata and with SasView's loader
# ----------------------------------------------------------------------------------------------------------------------

round_trip_cases = [
    (filename, exporter, extension)
    for filename in ONE_DIM_FILES
    for exporter, extension in [(NXcanSASExporter, "h5"), (CanSASXmlExporter, "xml"), (AsciiExporter, "txt"),
                                (CsvExporter, "csv")]
] + [
    (filename, exporter, extension)
    for filename in TWO_DIM_FILES
    for exporter, extension in [(NXcanSASExporter, "h5"), (AsciiExporter, "dat")]
] + [
    (filename, NXcanSASExporter, "h5") for filename in MIXED_FILES
]


@pytest.mark.parametrize("filename, exporter, extension", round_trip_cases)
def test_round_trip(tmp_path, filename, exporter, extension):
    data = load(filename)
    if not exporter.multiple_datasets:
        data = dict(list(data.items())[:1])
    destination = tmp_path / f"exported.{extension}"
    assert Exporter().export_data({destination: data}) == []

    assert file_format(destination) == EXPECTED_FORMAT[exporter]
    imported = import_file(destination)
    assert len(imported) == len(data)
    if exporter.multiple_datasets:
        # The keys (and so the names used by SasView) are kept
        assert list(imported.keys()) == list(data.keys())
    for original, result in zip(data.values(), imported.values()):
        assert_same_data(original, result)


@pytest.mark.parametrize("filename, exporter, extension", round_trip_cases)
def test_sasview_can_load(tmp_path, filename, exporter, extension):
    data = load(filename)
    if not exporter.multiple_datasets:
        data = dict(list(data.items())[:1])
    if any(len(np.ravel(d["I"].value)) < 5 for d in data.values()):
        pytest.skip("SasView will not load data with fewer than 5 points")
    for d in data.values():
        for name, unit in {"Q": units.per_angstrom, "Qx": units.per_angstrom, "Qy": units.per_angstrom,
                           "I": units.per_centimeter}.items():
            if name in d._data_contents and d[name].units != unit:
                pytest.skip(f"{name} is in {d[name].units}, and SasView only reads data in 1/Å and 1/cm correctly")
    destination = tmp_path / f"exported.{extension}"
    assert Exporter().export_data({destination: data}) == []
    loaded = legacy_load(destination)
    assert len(loaded) == len(data)
    for original, legacy in zip(data.values(), loaded):
        assert_legacy_matches(original, legacy)


@pytest.mark.parametrize("extension", ["h5", "xml", "txt", "csv"])
def test_synthetic_1d(tmp_path, extension):
    """Data in SasView's units is read correctly by SasView, and Q resolutions are kept"""
    original = make_1d(q_units=units.per_angstrom)
    destination = tmp_path / f"synthetic.{extension}"
    Exporter.export(original, destination)
    assert_same_data(original, first(import_file(destination)))
    legacy = legacy_load(destination)[0]
    assert_legacy_matches(original, legacy)
    assert np.allclose(legacy.dx, original["Q"].standard_error.value)


@pytest.mark.parametrize("extension", ["h5", "xml", "txt", "csv"])
def test_native_units_are_kept(tmp_path, extension):
    """Data is exported in its own units, without conversion to the units SasView assumes (1/Å and 1/cm)"""
    original = make_1d()
    assert original["Q"].units == units.per_nanometer
    destination = tmp_path / f"native.{extension}"
    Exporter.export(original, destination)
    result = first(import_file(destination))
    assert result["Q"].units == units.per_nanometer
    assert result["I"].units == units.per_centimeter
    # The numbers must be exactly as they were: not rescaled to 1/Å
    assert np.allclose(result["Q"].value, original["Q"].value)
    assert np.allclose(result["Q"].standard_error.value, original["Q"].standard_error.value)
    assert np.allclose(result["I"].value, original["I"].value)
    assert np.allclose(result["I"].standard_error.value, original["I"].standard_error.value)
    # SasView is deliberately not used to read these files: it only converts units it knows, and does not know nm^-1
    assert_same_data(original, result)


@pytest.mark.parametrize("extension", ["h5", "xml"])
def test_slit_resolution(tmp_path, extension):
    original = make_1d(slit=True)
    destination = tmp_path / f"slit.{extension}"
    Exporter.export(original, destination)
    assert_same_data(original, first(import_file(destination)))
    legacy = legacy_load(destination)[0]
    assert np.allclose(legacy.dxw, original["dQw"].value)
    assert np.allclose(legacy.dxl, original["dQl"].value)


@pytest.mark.parametrize("extension", ["h5", "dat"])
def test_synthetic_2d(tmp_path, extension):
    """Flat lists of 2D points on a grid are written so SasView recognises them as 2D data"""
    original = make_2d()
    destination = tmp_path / f"synthetic.{extension}"
    Exporter.export(original, destination)
    assert_same_data(original, first(import_file(destination)))
    legacy = legacy_load(destination)[0]
    assert_legacy_matches(original, legacy)
    assert np.allclose(legacy.err_data, original["I"].standard_error.value)


@pytest.mark.parametrize("extension", ["h5", "xml"])
def test_multiple_datasets(tmp_path, extension):
    data = [make_1d(name, q_units=units.per_angstrom) for name in ["first", "second", "first"]]
    destination = tmp_path / f"multiple.{extension}"
    Exporter.export(data, destination)
    imported = import_file(destination)
    assert list(imported.keys()) == ["first", "second", "first (2)"]
    for original, result in zip(data, imported.values()):
        assert_same_data(original, result)
    assert len(legacy_load(destination)) == 3


@pytest.mark.parametrize("filename", ["ISIS_1_1.xml", "cansas1d.xml", "cansas1d_units.xml", "valid_cansas_xml.xml"])
@pytest.mark.parametrize("extension", ["h5", "xml"])
def test_metadata_round_trip(tmp_path, filename, extension):
    data = load(filename)
    destination = tmp_path / f"metadata.{extension}"
    Exporter.export(data, destination)
    for original, result in zip(data.values(), import_file(destination).values()):
        assert result.metadata.title == original.metadata.title
        assert result.metadata.run == original.metadata.run
        assert result.metadata.sample == original.metadata.sample
        assert result.metadata.instrument == original.metadata.instrument
        assert result.metadata.process == original.metadata.process
        # NXcanSAS files must declare their definition
        assert result.metadata.definition in (original.metadata.definition, "NXcanSAS")


@pytest.mark.parametrize("filename", ONE_DIM_FILES)
def test_xml_is_valid_cansas(tmp_path, filename):
    """Exported XML must be valid against the canSAS 1.1 schema, even when metadata is missing"""
    destination = tmp_path / "exported.xml"
    Exporter.export(load(filename), destination)
    schema = etree.XMLSchema(etree.parse(str(SCHEMA)))
    document = etree.parse(str(destination))
    assert schema.validate(document), schema.error_log


def test_ascii_headings(tmp_path):
    """The headings give the units of the data as they are, not those SasView assumes"""
    destination = tmp_path / "headings.txt"
    Exporter.export(make_1d(), destination)
    lines = destination.read_text().splitlines()
    data_start = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    assert lines[data_start] == "Q(nm^-1) I(cm^-1) dI(cm^-1) dQ(nm^-1)"
    assert len(lines[data_start + 1].split()) == 4


def test_ascii_headings_in_sasview_units(tmp_path):
    destination = tmp_path / "headings.txt"
    Exporter.export(make_1d(q_units=units.per_angstrom), destination)
    lines = destination.read_text().splitlines()
    assert "Q(Ang^-1) I(cm^-1) dI(cm^-1) dQ(Ang^-1)" in lines


def test_csv_headings(tmp_path):
    destination = tmp_path / "headings.csv"
    Exporter.export(make_1d(), destination)
    lines = destination.read_text().splitlines()
    assert "Q(nm^-1),I(cm^-1),dI(cm^-1),dQ(nm^-1)" in lines
    assert len(lines[-1].split(",")) == 4
