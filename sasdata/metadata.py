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
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import h5py
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

    def as_h5(self, f: h5py.Group):
        """Export data onto an HDF5 group"""
        if self.x:
            self.x.as_h5(f, "x")
        if self.y:
            self.y.as_h5(f, "y")
        if self.z:
            self.z.as_h5(f, "z")


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

    def as_h5(self, f: h5py.Group):
        """Export data onto an HDF5 group"""
        if self.roll:
            self.roll.as_h5(f, "roll")
        if self.pitch:
            self.pitch.as_h5(f, "pitch")
        if self.yaw:
            self.yaw.as_h5(f, "yaw")


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

    def as_h5(self, group: h5py.Group):
        """Export data onto an HDF5 group"""
        if self.name is not None:
            group.create_dataset("name", data=[self.name])
        if self.distance:
            self.distance.as_h5(group, "SDD")
        if self.offset:
            self.offset.as_h5(group.create_group("offset"))
        if self.orientation:
            self.orientation.as_h5(group.create_group("orientation"))
        if self.beam_center:
            self.beam_center.as_h5(group.create_group("beam_center"))
        if self.pixel_size:
            self.pixel_size.as_h5(group.create_group("pixel_size"))
        if self.slit_length:
            self.slit_length.as_h5(group, "slit_length")

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


    def as_h5(self, group: h5py.Group):
        """Export data onto an HDF5 group"""
        if self.distance is not None:
            self.distance.as_h5(group, "distance")
        if self.name is not None:
            group.attrs["name"] = self.name
        if self.type_ is not None:
            group.attrs["type"] = self.type_
        if self.size:
            size_group = group.create_group("size")
            self.size.as_h5(size_group)
            if self.size_name is not None:
                size_group.attrs["name"] = self.size_name



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

    def as_h5(self, group: h5py.Group):
        """Export data onto an HDF5 group"""
        if self.length:
            self.length.as_h5(group, "length")
        for idx, a in enumerate(self.apertures):
            a.as_h5(group.create_group(f"sasaperture{idx:02d}"))


@dataclass(kw_only=True)
class BeamSize:
    name: str | None
    size: Vec3 | None

    @staticmethod
    def from_json(obj):
        return BeamSize(name=obj["name"], size=Vec3.from_json(obj["size"]))

    def as_h5(self, group: h5py.Group):
        """Export data onto an HDF5 group"""
        if self.name:
            group.attrs["name"] = self.name
        if self.size:
            self.size.as_h5(group)


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

    def as_h5(self, group: h5py.Group):
        """Export data onto an HDF5 group"""
        if self.radiation:
            group.create_dataset("radiation", data=[self.radiation])
        if self.beam_shape:
            group.create_dataset("beam_shape", data=[self.beam_shape])
        if self.beam_size:
            self.beam_size.as_h5(group.create_group("beam_size"))
        if self.wavelength:
            self.wavelength.as_h5(group, "wavelength")
        if self.wavelength_min:
            self.wavelength_min.as_h5(group, "wavelength_min")
        if self.wavelength_max:
            self.wavelength_max.as_h5(group, "wavelength_max")
        if self.wavelength_spread:
            self.wavelength_spread.as_h5(group, "wavelength_spread")





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

    def as_h5(self, f: h5py.Group):
        """Export data onto an HDF5 group"""
        if self.name is not None:
            f.attrs["name"] = self.name
        if self.sample_id is not None:
            f.create_dataset("ID", data=[self.sample_id])
        if self.thickness:
            self.thickness.as_h5(f, "thickness")
        if self.transmission is not None:
            f.create_dataset("transmission", data=[self.transmission])
        if self.temperature:
            self.temperature.as_h5(f, "temperature")
        if self.position:
            self.position.as_h5(f.create_group("position"))
        if self.orientation:
            self.orientation.as_h5(f.create_group("orientation"))
        if self.details:
            f.create_dataset("details", data=self.details)


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

    def as_h5(self, group: h5py.Group):
        """Export data onto an HDF5 group"""
        if self.name is not None:
            group.create_dataset("name", data=[self.name])
        if self.date is not None:
            group.create_dataset("date", data=[self.date])
        if self.description is not None:
            group.create_dataset("description", data=[self.description])
        if self.terms:
            for idx, (term, value) in enumerate(self.terms.items()):
                node = group.create_group(f"term{idx:02d}")
                node.attrs["name"] = term
                if type(value) is Quantity:
                    node.attrs["value"] = value.value
                    node.attrs["unit"] = value.units.symbol
                else:
                    node.attrs["value"] = value
        for idx, note in enumerate(self.notes):
            group.create_dataset(f"note{idx:02d}", data=[note])


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

    def as_h5(self, group: h5py.Group):
        """Export data onto an HDF5 group"""
        if self.source:
            self.source.as_h5(group.create_group("sassource"))
        for idx, c in enumerate(self.collimations):
            c.as_h5(group.create_group(f"sascollimation{idx:02d}"))
        for idx, d in enumerate(self.detector):
            d.as_h5(group.create_group(f"sasdetector{idx:02d}"))


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


@dataclass(kw_only=True, eq=True)
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

    def as_h5(self, f: h5py.Group):
        """Export data onto an HDF5 group"""
        for idx, run in enumerate(self.run):
            f.create_dataset(f"run{idx:02d}", data=[run])
        if self.title is not None:
            f.create_dataset("title", data=[self.title])
        if self.definition is not None:
            f.create_dataset("definition", data=[self.definition])
        if self.process:
            for idx, process in enumerate(self.process):
                name = f"sasprocess{idx:02d}"
                process.as_h5(f.create_group(name))
        if self.sample:
            self.sample.as_h5(f.create_group("sassample"))
        if self.instrument:
            self.instrument.as_h5(f.create_group("sasinstrument"))
        # self.raw.as_h5(meta) if self.raw else None


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
    """Use a string accessor to locate a key from within the data
    object.

    The basic grammar of these accessors explicitly match the python
    syntax for accessing the data.  For example, to access the `name`
    field within the object `person`, you would call
    `access_meta(person, ".name")`.  Similarly, lists and dicts are
    access with square brackets.

    > assert access_meta(person, '.name') == person.name
    > assert access_meta(person, '.phone.home') == person.phone.home
    > assert access_meta(person, '.addresses[0].postal_code') == person.address[0].postal_code
    > assert access_meta(person, '.children["Taylor"]') == person.children["Taylor"]

    Obviously, when the accessor is know ahead of time, `access_meta`
    provides no benefit over directly retrieving the data. However,
    when a data structure is loaded at runtime (e.g. the metadata of a
    neutron scattering file), then it isn't possible to know in
    advance the location of the specific value that the user desires.
    `access_meta` allows the user to provide the location at runtime.

    This function returns `None` when the key is not a valid address
    for any data within the structure.  Since the leaf could be any
    type that is not a list, dict, or dataclass, the return type of
    the function is `Any | None`.

    The list of locations within a structure is given by the
    `meta_tags` function.

    """
    result = obj
    while key != "":
        match key:
            case accessor if accessor.startswith("."):
                for fld in fields(result):
                    field_string = f".{fld.name}"
                    if accessor.startswith(field_string):
                        key = accessor[len(field_string) :]
                        result = getattr(result, fld.name)
                        break
            case index if (type(result) is list) and (matches := re.match(r"\[(\d+?)\](.*)", index)):
                result = result[int(matches[1])]
                key = matches[2]
            case name if (type(result) is dict) and (matches := re.match(r'\["(.+)"\](.*)', name)):
                result = result[matches[1]]
                key = matches[2]
            case _:
                return None
    return result


def meta_tags(obj: dataclass) -> list[str]:
    """Find all leaf accessors from a data object.

    The function treats the passed in object as a tree.  Lists, dicts,
    and dataclasses are all treated as branches on the tree and any
    other type is treated as a leaf.  The function then returns a list
    of strings, where each string is a "path" from the root of the
    tree to one leaf.  The structure of the path is designed to mimic
    the python code to access that specific leaf value.

    These accessors allow us to treat accessing entries within a
    structure as first class values.  This list can then be presented
    to the user to allow them to select specific information within
    the larger structure.  This is particularly important when plotting
    against a specific date value within the structure.

    Example:

    >@dataclass
     class Thermometer:
       temperature: float
       units: str
       params: list
    > item = Example()
    > item.temperature = 273
    > item.units = "K"
    > item.old_values = [{'date': '2025-08-12', 'temperature': 300'}]
    > assert meta_tags(item) = ['.temperature', '.units', '.old_values[0]["date"]', '.old_values[0]["temperature"]']

    The actual value of the leaf object specified by a path can be
    retrieved with the `access_meta` function.

    """
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
                for fld in fields(item):
                    items.append((f"{path}.{fld.name}", getattr(item, fld.name)))
            case _:
                result.append(path)
    return result


@dataclass(kw_only=True)
class TagCollection:
    """The collected tags and their variability."""

    singular: set[str] = field(default_factory=set)
    variable: set[str] = field(default_factory=set)


def collect_tags(objs: list[dataclass]) -> TagCollection:
    """Identify uniform and varying data within a groups of data objects

    The resulting TagCollection contains every accessor string that is
    valid for every object in the `objs` list.  For example, if
    `obj.name` is a string for every `obj` in `objs`, then the string
    ".name" will be present in one of the two sets in the tags
    collection.

    To be more specific, if `obj.name` exists and has the same value
    for every `obj` in `objs`, the string ".name" will be included in
    the `singular` set.  If there are at least two distinct values for
    `obj.name`, then ".name" will be in the `variable` set.

    """
    if not objs:
        return ([], [])
    first = objs.pop()
    terms = set(meta_tags(first))
    for obj in objs:
        terms = terms.intersection(set(meta_tags(obj)))

    objs.append(first)

    result = TagCollection()

    for term in terms:
        values = set([access_meta(obj, term) for obj in objs])
        if len(values) == 1:
            result.singular.add(term)
        else:
            result.variable.add(term)

    return result
