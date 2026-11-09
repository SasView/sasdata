"""
Export SasData to canSAS 1D XML files (version 1.1).

The files are valid against the canSAS 1.1 schema (http://www.cansas.org/formats/1.1/cansas1d.xsd), so they can be
read by both :class:`sasdata.data_io.importers.import_xml.XmlImporter` and SasView. The format only holds 1D data.
"""

import math
from pathlib import Path

from lxml import etree

from sasdata.data import SasMeasurement
from sasdata.data_io.exporter import Exporter, data_contents
from sasdata.dataset_types import one_dim
from sasdata.metadata import (
    Aperture,
    BeamSize,
    Collimation,
    Detector,
    Instrument,
    Metadata,
    Process,
    Rot3,
    Sample,
    Source,
    Vec3,
)
from sasdata.quantities.quantity import Quantity

CANSAS_VERSION = "1.1"
CANSAS_NAMESPACE = "urn:cansas1d:1.1"
CANSAS_SCHEMA = "http://www.cansas.org/formats/1.1/cansas1d.xsd"
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"


def tag(name: str) -> str:
    """The fully qualified name of a canSAS element"""
    return f"{{{CANSAS_NAMESPACE}}}{name}"


def format_float(value) -> str:
    """Format a number using the XML schema float representation"""
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "INF" if value > 0 else "-INF"
    return repr(value)


def sub_element(parent: etree._Element, element_name: str, text: str | None = None, /, **attrs) -> etree._Element:
    """Add a child element, with optional text and attributes (attributes with a value of None are skipped)"""
    element = etree.SubElement(parent, tag(element_name), {k: str(v) for k, v in attrs.items() if v is not None})
    if text is not None:
        element.text = text
    return element


def quantity_element(parent: etree._Element, element_name: str, quantity: Quantity | None, /, **attrs):
    """Add an element holding a single value with units, if the quantity is present"""
    if quantity is None:
        return None
    value = quantity.value
    if not isinstance(value, (int, float)):
        value = value.flat[0] if hasattr(value, "flat") else value[0]
    return sub_element(parent, element_name, format_float(value), unit=quantity.units.ascii_symbol, **attrs)


def vec3_element(parent: etree._Element, element_name: str, vector: Vec3 | None, name: str | None = None):
    if vector is None:
        return
    element = sub_element(parent, element_name, name=name)
    quantity_element(element, "x", vector.x)
    quantity_element(element, "y", vector.y)
    quantity_element(element, "z", vector.z)


def rot3_element(parent: etree._Element, element_name: str, rotation: Rot3 | None):
    if rotation is None:
        return
    element = sub_element(parent, element_name)
    quantity_element(element, "roll", rotation.roll)
    quantity_element(element, "pitch", rotation.pitch)
    quantity_element(element, "yaw", rotation.yaw)


class CanSASXmlExporter(Exporter):
    """Exporter for canSAS 1D XML (v1.1) files. Supports 1D data only, with any number of datasets per file."""

    name = "canSAS XML"
    aliases = ("cansas", "xml", "canSAS 1D XML")
    extensions = (".xml",)
    supported_dataset_types = (one_dim,)
    multiple_datasets = True

    def write(self, data: dict[str, SasMeasurement], path: Path):
        root = self.build_tree(data)
        etree.ElementTree(root).write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)

    def build_tree(self, data: dict[str, SasMeasurement]) -> etree._Element:
        """Build the XML tree for a list of datasets"""
        root = etree.Element(tag("SASroot"), nsmap={None: CANSAS_NAMESPACE, "xsi": XSI_NAMESPACE})
        root.set("version", CANSAS_VERSION)
        root.set(f"{{{XSI_NAMESPACE}}}schemaLocation", f"{CANSAS_NAMESPACE} {CANSAS_SCHEMA}")
        for name, datum in data.items():
            self.write_entry(root, name, datum)
        return root

    def write_entry(self, root: etree._Element, name: str, datum: SasMeasurement):
        """Write a single dataset to an SASentry, in the order required by the canSAS schema"""
        metadata: Metadata = datum.metadata
        entry = sub_element(root, "SASentry", name=name)
        sub_element(entry, "Title", metadata.title if metadata.title is not None else "")
        for run in metadata.run or [""]:
            sub_element(entry, "Run", run)
        self.write_data(entry, datum)
        self.write_sample(entry, metadata.sample)
        self.write_instrument(entry, metadata.instrument)
        for process in metadata.process or []:
            self.write_process(entry, process)
        # At least one SASnote is required by the schema.
        sub_element(entry, "SASnote")

    # ------------------------------------------------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def write_data(entry: etree._Element, datum: SasMeasurement):
        contents = data_contents(datum)
        q = contents["Q"]
        intensity = contents["I"]
        length = len(q.value)

        # Each column is (element name, values, unit). They are written in the order required by the schema.
        columns: list[tuple[str, object, str | None]] = [
            ("Q", q.value, q.units.ascii_symbol),
            ("I", intensity.value, intensity.units.ascii_symbol),
        ]
        if intensity.has_error:
            columns.append(("Idev", intensity.standard_error.value, intensity.units.ascii_symbol))
        # The schema allows either Qdev, or dQw and dQl.
        if q.has_error:
            columns.append(("Qdev", q.standard_error.value, q.units.ascii_symbol))
        elif "dQ" in contents:
            columns.append(("Qdev", contents["dQ"].in_units_of(q.units), q.units.ascii_symbol))
        else:
            for slit in ("dQw", "dQl"):
                if slit in contents:
                    columns.append((slit, contents[slit].value, contents[slit].units.ascii_symbol))
        if "Qmean" in contents:
            columns.append(("Qmean", contents["Qmean"].value, contents["Qmean"].units.ascii_symbol))
        if "Shadowfactor" in contents:
            # Shadowfactor is dimensionless and has no unit attribute in the schema.
            columns.append(("Shadowfactor", contents["Shadowfactor"].value, None))

        sasdata = sub_element(entry, "SASdata")
        for index in range(length):
            idata = sub_element(sasdata, "Idata")
            for column, values, unit in columns:
                sub_element(idata, column, format_float(values[index]), unit=unit)

    # ------------------------------------------------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def write_sample(entry: etree._Element, sample: Sample | None):
        element = sub_element(entry, "SASsample", name=sample.name if sample else None)
        # ID is required by the schema.
        sub_element(element, "ID", sample.sample_id if sample and sample.sample_id is not None else "")
        if sample is None:
            return
        quantity_element(element, "thickness", sample.thickness)
        if sample.transmission is not None:
            sub_element(element, "transmission", format_float(sample.transmission))
        quantity_element(element, "temperature", sample.temperature)
        vec3_element(element, "position", sample.position)
        rot3_element(element, "orientation", sample.orientation)
        for detail in sample.details or []:
            sub_element(element, "details", detail)

    def write_instrument(self, entry: etree._Element, instrument: Instrument | None):
        element = sub_element(entry, "SASinstrument")
        # The instrument name, source, a collimation and a detector are all required by the schema.
        sub_element(element, "name", "")
        self.write_source(element, instrument.source if instrument else None)
        for collimation in (instrument.collimations if instrument else None) or [None]:
            self.write_collimation(element, collimation)
        for detector in (instrument.detector if instrument else None) or [None]:
            self.write_detector(element, detector)

    @staticmethod
    def write_source(parent: etree._Element, source: Source | None):
        element = sub_element(parent, "SASsource")
        sub_element(element, "radiation", source.radiation if source and source.radiation is not None else "")
        if source is None:
            return
        beam_size: BeamSize | None = source.beam_size
        if beam_size is not None:
            vec3_element(element, "beam_size", beam_size.size or Vec3(x=None, y=None, z=None), beam_size.name)
        if source.beam_shape is not None:
            sub_element(element, "beam_shape", source.beam_shape)
        quantity_element(element, "wavelength", source.wavelength)
        quantity_element(element, "wavelength_min", source.wavelength_min)
        quantity_element(element, "wavelength_max", source.wavelength_max)
        quantity_element(element, "wavelength_spread", source.wavelength_spread)

    @staticmethod
    def write_collimation(parent: etree._Element, collimation: Collimation | None):
        element = sub_element(parent, "SAScollimation")
        if collimation is None:
            return
        quantity_element(element, "length", collimation.length)
        for aperture in collimation.apertures:
            aperture: Aperture
            aperture_element = sub_element(element, "aperture", name=aperture.name, type=aperture.type_)
            vec3_element(aperture_element, "size", aperture.size, aperture.size_name)
            quantity_element(aperture_element, "distance", aperture.distance)

    @staticmethod
    def write_detector(parent: etree._Element, detector: Detector | None):
        element = sub_element(parent, "SASdetector")
        sub_element(element, "name", detector.name if detector and detector.name is not None else "")
        if detector is None:
            return
        quantity_element(element, "SDD", detector.distance)
        vec3_element(element, "offset", detector.offset)
        rot3_element(element, "orientation", detector.orientation)
        vec3_element(element, "beam_center", detector.beam_center)
        vec3_element(element, "pixel_size", detector.pixel_size)
        quantity_element(element, "slit_length", detector.slit_length)

    @staticmethod
    def write_process(entry: etree._Element, process: Process):
        element = sub_element(entry, "SASprocess")
        if process.name is not None:
            sub_element(element, "name", process.name)
        if process.date is not None:
            sub_element(element, "date", process.date)
        if process.description is not None:
            sub_element(element, "description", process.description)
        for name, value in (process.terms or {}).items():
            if isinstance(value, Quantity):
                quantity_element(element, "term", value, name=name)
            else:
                sub_element(element, "term", str(value), name=name)
        # At least one SASprocessnote is required by the schema.
        for note in process.notes or [""]:
            sub_element(element, "SASprocessnote", note)
