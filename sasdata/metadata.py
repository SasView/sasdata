"""
Contains classes describing the metadata for a scattering run

The metadata is structures around the CANSas format version 1.1, found at
https://www.cansas.org/formats/canSAS1d/1.1/doc/specification.html

Metadata from other file formats should be massaged to fit into the data classes presented here.
Any useful metadata which cannot be included in these classes represent a bug in the CANSas format.

"""

import base64
import json
from dataclasses import dataclass

from numpy import ndarray

from sasdata.quantities.quantity import Quantity



@dataclass(kw_only=True)
class Vec3:
    """A three-vector of measured quantities"""

    x: Quantity[float] | None
    y: Quantity[float] | None
    z: Quantity[float] | None


@dataclass(kw_only=True)
class Rot3:
    """A measured rotation in 3-space"""

    roll: Quantity[float] | None
    pitch: Quantity[float] | None
    yaw: Quantity[float] | None


@dataclass(kw_only=True)
class Detector:
    """
    Detector information
    """

    name: str | None
    distance: Quantity[float] | None
    offset: Vec3 | None
    orientation: Rot3 | None
    beam_center: Vec3 | None
    pixel_size: Vec3 | None
    slit_length: Quantity[float] | None

    def summary(self):
        return (
            f"Detector:\n"
            f"   Name:         {self.name}\n"
            f"   Distance:     {self.distance}\n"
            f"   Offset:       {self.offset}\n"
            f"   Orientation:  {self.orientation}\n"
            f"   Beam center:  {self.beam_center}\n"
            f"   Pixel size:   {self.pixel_size}\n"
            f"   Slit length:  {self.slit_length}\n"
        )


@dataclass(kw_only=True)
class Aperture:
    distance: Quantity[float] | None
    size: Vec3 | None
    size_name: str | None
    name: str | None
    type_: str | None

    def summary(self):
        return (
            f"   Aperture:\n"
            f"     Name: {self.name}\n"
            f"     Aperture size: {self.size}\n"
            f"     Aperture distance: {self.distance}\n"
        )


@dataclass(kw_only=True)
class Collimation:
    """
    Class to hold collimation information
    """

    length: Quantity[float] | None
    apertures: list[Aperture]

    def summary(self):
        return f"Collimation:\n   Length: {self.length}\n" + "".join([a.summary() for a in self.apertures])


@dataclass(kw_only=True)
class BeamSize:
    name: str | None
    size: Vec3 | None


@dataclass(kw_only=True)
class Source:
    radiation: str | None
    beam_shape: str | None
    beam_size: BeamSize | None
    wavelength: Quantity[float] | None
    wavelength_min: Quantity[float] | None
    wavelength_max: Quantity[float] | None
    wavelength_spread: Quantity[float] | None

    def summary(self) -> str:
        return (
            f"Source:\n"
            f"    Radiation:         {self.radiation}\n"
            f"    Shape:             {self.beam_shape}\n"
            f"    Wavelength:        {self.wavelength}\n"
            f"    Min. Wavelength:   {self.wavelength_min}\n"
            f"    Max. Wavelength:   {self.wavelength_max}\n"
            f"    Wavelength Spread: {self.wavelength_spread}\n"
            f"    Beam Size:         {self.beam_size}\n"
        )


@dataclass(kw_only=True)
class Sample:
    """
    Class to hold the sample description
    """

    name: str | None
    sample_id: str | None
    thickness: Quantity[float] | None
    transmission: float | None
    temperature: Quantity[float] | None
    position: Vec3 | None
    orientation: Rot3 | None
    details: list[str]

    def summary(self) -> str:
        return (
            f"Sample:\n"
            f"   ID:           {self.sample_id}\n"
            f"   Transmission: {self.transmission}\n"
            f"   Thickness:    {self.thickness}\n"
            f"   Temperature:  {self.temperature}\n"
            f"   Position:     {self.position}\n"
            f"   Orientation:  {self.orientation}\n"
        )


@dataclass(kw_only=True)
class Process:
    """
    Class that holds information about the processes
    performed on the data.
    """

    name: str | None
    date: str | None
    description: str | None
    terms: dict[str, str | Quantity[float]]
    notes: list[str]

    def single_line_desc(self):
        """
        Return a single line string representing the process
        """
        return f"{self.name.value} {self.date.value} {self.description.value}"

    def summary(self):
        if self.terms:
            termInfo = "    Terms:\n" + "\n".join([f"        {k}: {v}" for k, v in self.terms.items()]) + "\n"
        else:
            termInfo = ""

        if self.notes:
            noteInfo = "    Notes:\n" + "\n".join([f"        {note}" for note in self.notes]) + "\n"
        else:
            noteInfo = ""

        return (
            f"Process:\n"
            f"    Name: {self.name}\n"
            f"    Date: {self.date}\n"
            f"    Description: {self.description}\n"
            f"{termInfo}"
            f"{noteInfo}"
        )


@dataclass
class Instrument:
    collimations: list[Collimation]
    source: Source | None
    detector: list[Detector]

    def summary(self):
        return (
            "\n".join([c.summary() for c in self.collimations])
            + "".join([d.summary() for d in self.detector])
            + (self.source.summary() if self.source is not None else "")
        )


@dataclass(kw_only=True)
class MetaNode:
    name: str
    attrs: dict[str, str]
    contents: str | Quantity | ndarray | list["MetaNode"]

    def to_string(self, header=""):
        """Convert node to pretty printer string"""
        if self.attrs:
            attributes = f"\n{header}  Attributes:\n" + "\n".join(
                [f"{header}    {k}: {v}" for k, v in self.attrs.items()]
            )
        else:
            attributes = ""
        if self.contents:
            if type(self.contents) is str:
                children = f"\n{header}  {self.contents}"
            else:
                children = "".join([n.to_string(header + "  ") for n in self.contents])
        else:
            children = ""

        return f"\n{header}{self.name}:{attributes}{children}"

    def filter(self, name: str) -> list[ndarray | Quantity | str]:
        match self.contents:
            case str() | ndarray() | Quantity():
                if name == self.name:
                    return [self.contents]
            case list():
                return [y for x in self.contents for y in x.filter(name)]
            case _:
                raise RuntimeError(f"Cannot filter contents of type {type(self.contents)}: {self.contents}")
        return []


@dataclass(kw_only=True)
class Metadata:
    title: str | None
    run: list[str]
    definition: str | None
    process: list[Process]
    sample: Sample | None
    instrument: Instrument | None
    raw: MetaNode

    def summary(self):
        run_string = self.run[0] if len(self.run) == 1 else self.run
        return (
            f"  {self.title}, Run: {run_string}\n"
            + "  "
            + "=" * len(self.title if self.title else "")
            + "======="
            + "=" * len(run_string)
            + "\n\n"
            + f"Definition: {self.title}\n"
            + "".join([p.summary() for p in self.process])
            + (self.sample.summary() if self.sample else "")
            + (self.instrument.summary() if self.instrument else "")
        )


class MetadataEncoder(json.JSONEncoder):
    def default(self, obj):
        match obj:
            case None:
                return None
            case Quantity():
                return None
            case ndarray():
                return {
                    "type": "ndarray",
                    "encoding": "base64",
                    "contents": base64.b64encode(obj.dumps()).decode("utf-8"),
                }
            case Vec3():
                return {
                    "x": self.default(obj.x),
                    "y": self.default(obj.y),
                    "z": self.default(obj.z),
                }
            case Rot3():
                return {
                    "roll": self.default(obj.roll),
                    "pitch": self.default(obj.pitch),
                    "yaw": self.default(obj.yaw),
                }
            case Sample():
                return {
                    "name": obj.name,
                    "sample_id": obj.sample_id,
                    "thickness": obj.thickness,
                    "transmission": obj.transmission,
                    "position": self.default(obj.position),
                    "orientation": self.default(obj.orientation),
                    "details": obj.details,
                }
            case Process():
                return {
                    "name": obj.name,
                    "date": obj.date,
                    "description": obj.description,
                    "terms": {k: self.default(obj.terms[k]) for k in obj.terms},
                    "nodes": obj.notes,
                }
            case Aperture():
                return {
                    "distance": self.default(obj.distance),
                    "size": self.default(obj.size),
                    "size_name": obj.size_name,
                    "name": obj.name,
                    "type": obj.type_,
                }
            case Collimation():
                return {
                    "length": self.default(obj.length),
                    "apertures": [self.default(a) for a in obj.apertures],
                }
            case BeamSize():
                return {"name": obj.name, "size": self.default(obj.size)}
            case Source():
                return {
                    "radiation": obj.radiation,
                    "beam_shape": obj.beam_shape,
                    "beam_size": self.default(obj.beam_size),
                    "wavelength": self.default(obj.wavelength),
                    "wavelength_min": self.default(obj.wavelength_min),
                    "wavelength_max": self.default(obj.wavelength_max),
                    "wavelength_spread": self.default(obj.wavelength_spread),
                }
            case Detector():
                return {
                    "name": obj.name,
                    "distance": self.default(obj.distance),
                    "offset": self.default(obj.offset),
                    "orientation": self.default(obj.orientation),
                    "beam_center": self.default(obj.beam_center),
                    "pixel_size": self.default(obj.pixel_size),
                    "slit_length": self.default(obj.slit_length),
                }
            case Instrument():
                return {
                    "collimations": [self.default(c) for c in obj.collimations],
                    "source": self.default(obj.source),
                    "detector": [self.default(d) for d in obj.detector],
                }
            case MetaNode():
                return {"name": obj.name, "attrs": obj.attrs, "contents": obj.contents}
            case Metadata():
                return {
                    "title": obj.title,
                    "run": obj.run,
                    "definition": obj.definition,
                    "process": [self.default(p) for p in obj.process],
                    "sample": self.default(obj.sample),
                    "instrument": self.default(obj.instrument),
                    "raw": self.default(obj.raw),
                }
            case _:
                return super().default(obj)
