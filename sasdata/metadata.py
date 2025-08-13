"""
Contains classes describing the metadata for a scattering run

The metadata is structures around the CANSas format version 1.1, found at
https://www.cansas.org/formats/canSAS1d/1.1/doc/specification.html

Metadata from other file formats should be massaged to fit into the data classes presented here.
Any useful metadata which cannot be included in these classes represent a bug in the CANSas format.

"""

import base64
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np
from numpy import ndarray

from sasdata.quantities.quantity import Quantity
from sasdata.quantities.unit_parser import parse_unit
from sasdata.quantities.units import NamedUnit


def from_json_quantity(obj: dict) -> Quantity | None:
    if obj is None:
        return None

    return Quantity(obj["value"], parse_unit(obj["units"]))


@dataclass(kw_only=True)
class Vec3:
    """A three-vector of measured quantities"""

    x: Quantity[float] | None
    y: Quantity[float] | None
    z: Quantity[float] | None

    @staticmethod
    def from_json(obj: dict) -> Quantity | None:
        if obj is None:
            return None
        return Vec3(
            x=from_json_quantity(obj["x"]),
            y=from_json_quantity(obj["y"]),
            z=from_json_quantity(obj["z"]),
        )


@dataclass(kw_only=True)
class Rot3:
    """A measured rotation in 3-space"""

    roll: Quantity[float] | None
    pitch: Quantity[float] | None
    yaw: Quantity[float] | None

    @staticmethod
    def from_json(obj: dict) -> Quantity | None:
        if obj is None:
            return None
        return Vec3(
            roll=from_json_quantity(obj["roll"]),
            pitch=from_json_quantity(obj["pitch"]),
            yaw=from_json_quantity(obj["yaw"]),
        )


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

    @staticmethod
    def from_json(obj):
        return Detector(
            name=obj["name"],
            distance=from_json_quantity(obj["distance"]),
            offset=Vec3.from_json(obj["offset"]),
            orientation=Rot3.from_json(obj["orientation"]),
            beam_center=Vec3.from_json(obj["beam_center"]),
            pixel_size=Vec3.from_json(obj["pixel_size"]),
            slit_length=from_json_quantity(obj["slit_length"]),
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

    @staticmethod
    def from_json(obj):
        return Aperture(
            distance=from_json_quantity(obj["distance"]),
            size=Vec3.from_json(obj["size"]),
            size_name=obj["size_name"],
            name=obj["name"],
            type_=obj["type_"],
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

    @staticmethod
    def from_json(obj):
        return Collimation(
            length=from_json_quantity(obj["length"]) if obj["length"] else None,
            apertures=list(map(Aperture.from_json, obj["apertures"])),
        )


@dataclass(kw_only=True)
class BeamSize:
    name: str | None
    size: Vec3 | None

    @staticmethod
    def from_json(obj):
        return BeamSize(name=obj["name"], size=Vec3.from_json(obj["size"]))


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

    @staticmethod
    def from_json(obj):
        return Source(
            radiation=obj["radiation"],
            beam_shape=obj["beam_shape"],
            beam_size=BeamSize.from_json(obj["beam_size"]) if obj["beam_size"] else None,
            wavelength=obj["wavelength"],
            wavelength_min=obj["wavelength_min"],
            wavelength_max=obj["wavelength_max"],
            wavelength_spread=obj["wavelength_spread"],
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

    @staticmethod
    def from_json(obj):
        return Sample(
            name=obj["name"],
            sample_id=obj["sample_id"],
            thickness=obj["thickness"],
            transmission=obj["transmission"],
            temperature=obj["temperature"],
            position=obj["position"],
            orientation=obj["orientation"],
            details=obj["details"],
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

    @staticmethod
    def from_json(obj):
        return Process(
            name=obj["name"],
            date=obj["date"],
            description=obj["description"],
            terms=obj["terms"],
            notes=obj["notes"],
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

    @staticmethod
    def from_json(obj):
        return Instrument(
            collimations=list(map(Collimation.from_json, obj["collimations"])),
            source=Source.from_json(obj["source"]),
            detector=list(map(Detector.from_json, obj["detector"])),
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

    def __eq__(self, other) -> bool:
        """Custom equality overload needed since numpy arrays don't
        play nicely with equality"""
        match self.contents:
            case ndarray():
                if not np.all(self.contents == other.contents):
                    return False
            case Quantity():
                result = self.contents == other.contents
                if type(result) is ndarray and not np.all(result):
                    return False
                if type(result) is bool and not result:
                    return False
            case _:
                print(type(self.contents))
                if self.contents != other.contents:
                    return False
        for k, v in self.attrs.items():
            if k not in other.attrs:
                return False
            if type(v) is np.ndarray and np.any(v != other.attrs[k]):
                return False
            if type(v) is not np.ndarray and v != other.attrs[k]:
                return False
        return self.name == other.name

    @staticmethod
    def from_json(obj):
        def from_content(con):
            match con:
                case list():
                    return list(map(MetaNode.from_json, con))
                case {
                    "type": "ndarray",
                    "dtype": dtype,
                    "encoding": "base64",
                    "contents": contents,
                    "shape": shape,
                }:
                    return np.frombuffer(base64.b64decode(contents), dtype=dtype).reshape(shape)
                case {"value": value, "units": units}:
                    return from_json_quantity({"value": from_content(value), "units": from_content(units)})
                case _:
                    return con

        return MetaNode(
            name=obj["name"],
            attrs={k: from_content(v) for k, v in obj["attrs"].items()},
            contents=from_content(obj["contents"]),
        )


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

    @staticmethod
    def from_json(obj):
        return Metadata(
            title=obj["title"],
            run=obj["run"],
            definition=obj["definition"],
            process=[Process.from_json(p) for p in obj["process"]],
            sample=Sample.from_json(obj["sample"]),
            instrument=Instrument.from_json(obj["instrument"]),
            raw=MetaNode.from_json(obj["raw"]),
        )


class MetadataEncoder(json.JSONEncoder):
    def default(self, obj):
        match obj:
            case None:
                return None
            case NamedUnit():
                return obj.name
            case Quantity():
                return {"value": obj.value, "units": obj.units}
            case ndarray():
                return {
                    "type": "ndarray",
                    "encoding": "base64",
                    "contents": base64.b64encode(obj.tobytes()).decode("utf-8"),
                    "dtype": obj.dtype.str,
                    "shape": obj.shape,
                }
            case Vec3():
                return {
                    "x": obj.x,
                    "y": obj.y,
                    "z": obj.z,
                }
            case Rot3():
                return {
                    "roll": obj.roll,
                    "pitch": obj.pitch,
                    "yaw": obj.yaw,
                }
            case Sample():
                return {
                    "name": obj.name,
                    "sample_id": obj.sample_id,
                    "thickness": obj.thickness,
                    "transmission": obj.transmission,
                    "temperature": obj.temperature,
                    "position": obj.position,
                    "orientation": obj.orientation,
                    "details": obj.details,
                }
            case Process():
                return {
                    "name": obj.name,
                    "date": obj.date,
                    "description": obj.description,
                    "terms": {k: obj.terms[k] for k in obj.terms},
                    "notes": obj.notes,
                }
            case Aperture():
                return {
                    "distance": obj.distance,
                    "size": obj.size,
                    "size_name": obj.size_name,
                    "name": obj.name,
                    "type": obj.type_,
                }
            case Collimation():
                return {
                    "length": obj.length,
                    "apertures": [a for a in obj.apertures],
                }
            case BeamSize():
                return {"name": obj.name, "size": obj.size}
            case Source():
                return {
                    "radiation": obj.radiation,
                    "beam_shape": obj.beam_shape,
                    "beam_size": obj.beam_size,
                    "wavelength": obj.wavelength,
                    "wavelength_min": obj.wavelength_min,
                    "wavelength_max": obj.wavelength_max,
                    "wavelength_spread": obj.wavelength_spread,
                }
            case Detector():
                return {
                    "name": obj.name,
                    "distance": obj.distance,
                    "offset": obj.offset,
                    "orientation": obj.orientation,
                    "beam_center": obj.beam_center,
                    "pixel_size": obj.pixel_size,
                    "slit_length": obj.slit_length,
                }
            case Instrument():
                return {
                    "collimations": [c for c in obj.collimations],
                    "source": obj.source,
                    "detector": [d for d in obj.detector],
                }
            case MetaNode():
                return {"name": obj.name, "attrs": obj.attrs, "contents": obj.contents}
            case Metadata():
                return {
                    "title": obj.title,
                    "run": obj.run,
                    "definition": obj.definition,
                    "process": [p for p in obj.process],
                    "sample": obj.sample,
                    "instrument": obj.instrument,
                    "raw": obj.raw,
                }
            case _:
                return super().default(obj)


def access_meta(obj: dataclass, key: str) -> Any | None:
    result = obj
    while key != "":
        match key:
            case accessor if accessor.startswith("."):
                for field in fields(result):
                    field_string = f".{field.name}"
                    if accessor.startswith(field_string):
                        key = accessor[len(field_string):]
                        result = getattr(result, field.name)
                        break
            case index if (type(result) is list) and (matches := re.match("\[(\d+?)\](.+)", index)):
                result = result[int(matches[1])]
                key = matches[2]
            case index if (type(result) is dict) and (matches := re.match('\["(.+?)"\](.+)', index)):
                result = result[matches[1]]
                key = matches[2]
    return result

def meta_tags(obj: dataclass) -> list[str]:
    result = []
    items = [("", obj)]
    while items:
        path, item = items.pop()
        match item:
            case list(xs):
                for idx, x in enumerate(xs):
                    items.append((f"{path}[{idx}]", x))
            case dict(xs):
                for k, v in xs.items():
                    items.append((f'{path}["{k}"]', v))
            case n if is_dataclass(n):
                for field in fields(item):
                    items.append((f"{path}.{field.name}", getattr(item, field.name)))
            case _:
                result.append(path)
    return result
