from tokenize import String

import numpy as np
from numpy.typing import ArrayLike

import sasdata.quantities.units as units
from sasdata.quantities.absolute_temperature import AbsoluteTemperatureAccessor
from sasdata.quantities.accessors import StringAccessor, LengthAccessor, AngleAccessor, QuantityAccessor, \
    DimensionlessAccessor, FloatAccessor, TemperatureAccessor, AccessorTarget


from dataclasses import dataclass

from sasdata.quantities.quantity import Quantity

@dataclass(kw_only=True)
class Vec3:
    """A three-vector of measured quantities"""
    x : Quantity[float] | None
    y : Quantity[float] | None
    z : Quantity[float] | None

    @staticmethod
    def deserialise_json(json_data: dict):
        x = None
        y = None
        z = None
        if "x" in json_data:
            x = Quantity.deserialise_json(json_data["x"])
        if "y" in json_data:
            y = Quantity.deserialise_json(json_data["y"])
        if "z" in json_data:
            z = Quantity.deserialise_json(json_data["z"])
        return Vec3(x=x, y=y, z=z)

    def serialise_json(self):
        data = {
            "x": None,
            "y": None,
            "z": None
        }
        if self.x is not None:
            data["x"] = self.x.serialise_json()
        if self.y is not None:
            data["y"] = self.y.serialise_json()
        if self.z is not None:
            data["z"] = self.z.serialise_json()
        return data

@dataclass(kw_only=True)
class Rot3:
    """A measured rotation in 3-space"""
    roll : Quantity[float] | None
    pitch : Quantity[float] | None
    yaw : Quantity[float] | None

    @staticmethod
    def deserialise_json(json_data: dict):
        roll = None
        pitch = None
        yaw = None
        if "roll" in json_data:
            roll = Quantity.deserialise_json(json_data["roll"])
        if "pitch" in json_data:
            pitch = Quantity.deserialise_json(json_data["pitch"])
        if "yaw" in json_data:
            yaw = Quantity.deserialise_json(json_data["yaw"])
        return Rot3(roll=roll, pitch=pitch, yaw=yaw)

    def serialise_json(self):
        data = {
            "roll": None,
            "pitch": None,
            "yaw": None
        }
        if self.roll is not None:
            data["roll"] = self.roll.serialise_json()
        if self.pitch is not None:
            data["pitch"] = self.pitch.serialise_json()
        if self.yaw is not None:
            data["yaw"] = self.yaw.serialise_json()
        return data

@dataclass(kw_only=True)
class Detector:
    """
    Detector information
    """

    def __init__(self, target_object: AccessorTarget):

        # Name of the instrument [string]
        self.name = StringAccessor(target_object, "name")

        # Sample to detector distance [float] [mm]
        self.distance = LengthAccessor[float](target_object,
                                              "distance",
                                              "distance.units",
                                              default_unit=units.millimeters)

        # Offset of this detector position in X, Y,
        # (and Z if necessary) [Vector] [mm]
        self.offset = LengthAccessor[ArrayLike](target_object,
                                                "offset",
                                                "offset.units",
                                                default_unit=units.millimeters)

        self.orientation = AngleAccessor[ArrayLike](target_object,
                                                    "orientation",
                                                    "orientation.units",
                                                    default_unit=units.degrees)

        self.beam_center = LengthAccessor[ArrayLike](target_object,
                                                     "beam_center",
                                                     "beam_center.units",
                                                     default_unit=units.millimeters)

        # Pixel size in X, Y, (and Z if necessary) [Vector] [mm]
        self.pixel_size = LengthAccessor[ArrayLike](target_object,
                                                    "pixel_size",
                                                    "pixel_size.units",
                                                    default_unit=units.millimeters)

        # Slit length of the instrument for this detector.[float] [mm]
        self.slit_length = LengthAccessor[float](target_object,
                                                 "slit_length",
                                                 "slit_length.units",
                                                 default_unit=units.millimeters)

    def summary(self):
        return (f"Detector:\n"
                f"   Name:         {self.name}\n"
                f"   Distance:     {self.distance}\n"
                f"   Offset:       {self.offset}\n"
                f"   Orientation:  {self.orientation}\n"
                f"   Beam center:  {self.beam_center}\n"
                f"   Pixel size:   {self.pixel_size}\n"
                f"   Slit length:  {self.slit_length}\n")

    @staticmethod
    def deserialise_json(json_data: dict):
        name = None
        distance = None
        offset = None
        orientation = None
        beam_center = None
        pixel_size = None
        slit_length = None
        if "name" in json_data:
            name = json_data["name"]
        if "distance" in json_data:
            distance = Quantity.deserialise_json(json_data["distance"])
        if "offset" in json_data:
            offset = Vec3.deserialise_json(json_data["offset"])
        if "orientation" in json_data:
            orientation = Rot3.deserialise_json(json_data["orientation"])
        if "beam_center" in json_data:
            beam_center = Vec3.deserialise_json(json_data["beam_center"])
        if "pixel_size" in json_data:
            pixel_size = Vec3.deserialise_json(json_data["pixel_size"])
        if "slit_length" in json_data:
            slit_length = Quantity.deserialise_json(json_data["slit_length"])
        return Detector(
            name=name,
            distance=distance,
            offset=offset,
            orientation=orientation,
            beam_center=beam_center,
            pixel_size=pixel_size,
            slit_length=slit_length
        )


    def serialise_json(self):
        data = {
            "name": self.name,
            "distance": None,
            "offset": None,
            "orientation": None,
            "beam_center": None,
            "pixel_size": None,
            "slit_length": None
        }
        if self.distance is not None:
            data["distance"] = self.distance.serialise_json()
        if self.offset is not None:
            data["offset"] = self.offset.serialise_json()
        if self.orientation is not None:
            data["orientation"] = self.orientation.serialise_json()
        if self.beam_center is not None:
            data["beam_center"] = self.beam_center.serialise_json()
        if self.pixel_size is not None:
            data["pixel_size"] = self.pixel_size.serialise_json()
        if self.slit_length is not None:
            data["slit_length"] = self.slit_length.serialise_json()
        return data


class Aperture:

    def __init__(self, target_object: AccessorTarget):

        # Name
        self.name = StringAccessor(target_object, "name")

        # Type
        self.type = StringAccessor(target_object, "type")

        # Size name - TODO: What is the name of a size
        self.size_name = StringAccessor(target_object, "size_name")

        # Aperture size [Vector] # TODO: Wat!?!
        self.size = QuantityAccessor[ArrayLike](target_object,
                                "size",
                                "size.units",
                                default_unit=units.millimeters)

        # Aperture distance [float]
        self.distance = LengthAccessor[float](target_object,
                                    "distance",
                                    "distance.units",
                                    default_unit=units.millimeters)


    def summary(self):
        return (f"Aperture:\n"
                f"  Name: {self.name}\n"
                f"  Aperture size: {self.size}\n"
                f"  Aperture distance: {self.distance}\n")

    @staticmethod
    def deserialise_json(json_data: dict):
        distance = None
        size = None
        size_name = None
        name = None
        type_ = None
        if "distance" in json_data:
            distance = Quantity.deserialise_json(json_data["distance"])
        if "size" in json_data:
            size = Vec3.deserialise_json(json_data["size"])
        if "size_name" in json_data:
            size_name = json_data["size_name"]
        if "name" in json_data:
            name = json_data["name"]
        if "type" in json_data:
            type_ = json_data["type"]
        return Aperture(
            distance=distance, size=size, size_name=size_name, name=name, type_=type_
        )

    def serialise_json(self):
        data = {
            "distance": None,
            "size": None,
            "size_name": self.size_name,
            "name": self.name,
            "type": self.type_
        }
        if self.distance is not None:
            data["distance"] = self.distance.serialise_json()
        if self.size is not None:
            data["size"] = self.size.serialise_json()

class Collimation:
    """
    Class to hold collimation information
    """

    def __init__(self, name, length):

        # Name
        self.name = name
        # Length [float] [mm]
        self.length = length
        # TODO - parse units properly

    def summary(self):

        #TODO collimation stuff
        return (
            f"Collimation:\n"
            f"   Length: {self.length}\n")

    @staticmethod
    def deserialise_json(json_data: dict):
        length = None
        apertures = []
        if "length" in json_data:
            length = Quantity.deserialise_json(json_data["length"])
        if "apertures" in json_data:
            apertures = [Aperture.deserialise_json(a) for a in json_data["apertures"]]

    def serialise_json(self):
        data = {
            "length": None,
            "apertures": [a.serialise_json() for a in self.apertures]
        }
        if self.length is not None:
            data["length"] = self.length.serialise_json()
        return data

@dataclass
class BeamSize:
    name: str | None
    size: Vec3 | None

    @staticmethod
    def deserialise_json(json_data: dict):
        name = None
        size = None
        if "name" in json_data:
            name = json_data["name"]
        if "size" in json_data:
            size = Vec3.deserialise_json(json_data["size"])
        return BeamSize(name=name, size=size)

    def serialise_json(self):
        data = {
            "name": self.name,
            "size": None
        }
        if self.size is not None:
            data["size"] = self.size.serialise_json()
        return data


@dataclass
class Source:
    radiation: str
    beam_shape: str
    beam_size: Optional[BeamSize]
    wavelength : Quantity[float]
    wavelength_min : Quantity[float]
    wavelength_max : Quantity[float]
    wavelength_spread : Quantity[float]

    def summary(self) -> str:
        if self.radiation is None and self.type.value and self.probe_particle.value:
            radiation = f"{self.type.value} {self.probe_particle.value}"
        else:
            radiation = f"{self.radiation}"

        return (
            f"Source:\n"
            f"    Radiation:         {radiation}\n"
            f"    Shape:             {self.beam_shape}\n"
            f"    Wavelength:        {self.wavelength}\n"
            f"    Min. Wavelength:   {self.wavelength_min}\n"
            f"    Max. Wavelength:   {self.wavelength_max}\n"
            f"    Wavelength Spread: {self.wavelength_spread}\n"
            f"    Beam Size:         {self.beam_size}\n"
        )

    @staticmethod
    def deserialise_json(json_data: dict):
        radiation = None
        beam_shape = None
        beam_size = None
        wavelength = None
        wavelength_min = None
        wavelength_max = None
        wavelength_spread = None
        if "radiation" in json_data:
            radiation = json_data["radiation"]
        if "beam_shape" in json_data:
            beam_shape = json_data["beam_shape"]
        if "beam_size" in json_data:
            beam_size = BeamSize.deserialise_json(json_data["beam_size"])
        if "wavelength" in json_data:
            wavelength = Quantity.deserialise_json(json_data["wavelength"])
        if "wavelength_min" in json_data:
            wavelength_min = Quantity.deserialise_json(json_data["wavelength_min"])
        if "wavelength_max" in json_data:
            wavelength_max = Quantity.deserialise_json(json_data["wavelength_max"])
        if "wavelength_spread" in json_data:
            wavelength_spread = Quantity.deserialise_json(json_data["wavelength_spread"])
        return Source(
            radiation=radiation,
            beam_shape=beam_shape,
            beam_size=beam_size,
            wavelength=wavelength,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            wavelength_spread=wavelength_spread
        )

    def serialise_json(self):
        data = {
            "radiation": self.radiation,
            "beam_shape": self.beam_shape,
            "beam_size": None,
            "wavelength": None,
            "wavelength_min": None,
            "wavelength_max": None,
            "wavelength_spread": None
        }
        if self.beam_size is not None:
            data["beam_size"] = self.beam_size.serialise_json()
        if self.wavelength is not None:
            data["wavelength"] = self.wavelength.serialise_json()
        if self.wavelength_min is not None:
            data["wavelength_min"] = self.wavelength_min.serialise_json()
        if self.wavelength_max is not None:
            data["wavelength_max"] = self.wavelength_max.serialise_json()
        if self.wavelength_spread is not None:
            data["wavelength_spread"] = self.wavelength_spread.serialise_json()
        return data


"""
Definitions of radiation types
"""
NEUTRON = 'neutron'
XRAY = 'x-ray'
MUON = 'muon'
ELECTRON = 'electron'


class Sample:
    """
    Class to hold the sample description
    """
    def __init__(self, target_object: AccessorTarget):

        # Short name for sample
        self.name = StringAccessor(target_object, "name")
        # ID

        self.sample_id = StringAccessor(target_object, "id")

        # Thickness [float] [mm]
        self.thickness = LengthAccessor(target_object,
                                        "thickness",
                                        "thickness.units",
                                        default_unit=units.millimeters)

        # Transmission [float] [fraction]
        self.transmission = FloatAccessor(target_object,"transmission")

        # Temperature [float] [No Default]
        self.temperature = AbsoluteTemperatureAccessor(target_object,
                                                       "temperature",
                                                       "temperature.unit",
                                                       default_unit=units.kelvin)
        # Position [Vector] [mm]
        self.position = LengthAccessor[ArrayLike](target_object,
                                                  "position",
                                                  "position.unit",
                                                  default_unit=units.millimeters)

        # Orientation [Vector] [degrees]
        self.orientation = AngleAccessor[ArrayLike](target_object,
                                                    "orientation",
                                                    "orientation.unit",
                                                    default_unit=units.degrees)

        # Details
        self.details = StringAccessor(target_object, "details")


        # SESANS zacceptance
        zacceptance = (0,"")
        yacceptance = (0,"")

    def summary(self) -> str:
        return (f"Sample:\n"
                f"   ID:           {self.sample_id}\n"
                f"   Transmission: {self.transmission}\n"
                f"   Thickness:    {self.thickness}\n"
                f"   Temperature:  {self.temperature}\n"
                f"   Position:     {self.position}\n"
                f"   Orientation:  {self.orientation}\n")

    @staticmethod
    def deserialise_json(json_data):
        name = None
        sample_id = None
        thickness = None
        transmission = None
        temperature = None
        position = None
        orientation = None
        details = []
        if "name" in json_data:
            name = json_data["name"]
        if "sample_id" in json_data:
            sample_id = json_data["sample_id"]
        if "thickness" in json_data:
            thickness = Quantity.deserialise_json(json_data["thickness"])
        if "temperature" in json_data:
            temperature = Quantity.deserialise_json(json_data["temperature"])
        if "position" in json_data:
            position = Vec3.deserialise_json(json_data["position"])
        if "orientation" in json_data:
            orientation = Rot3.deserialise_json(json_data["orientation"])
        return Sample(
            name=name,
            sample_id=sample_id,
            thickness=thickness,
            transmission=transmission,
            temperature=temperature,
            position=position,
            orientation=orientation,
            details=details
        )


    def serialise_json(self):
        data = {
            "name": self.name,
            "sample_id": self.sample_id,
            "thickness": None,
            "transmission": self.transmission,
            "temperature": None,
            "position": None,
            "orientation": None,
            "details": self.details
        }
        if self.thickness is not None:
            data["thickness"] = self.thickness.serialise_json()
        if self.temperature is not None:
            data["temperature"] = self.temperature.serialise_json()
        if self.position is not None:
            data["position"] = self.position.serialise_json()
        if self.orientation is not None:
            data["orientation"] = self.orientation.serialise_json()
        return data


class Process:
    """
    Class that holds information about the processes
    performed on the data.
    """
    def __init__(self, target_object: AccessorTarget):
        self.name = StringAccessor(target_object, "name")
        self.date = StringAccessor(target_object, "date")
        self.description = StringAccessor(target_object, "description")

        #TODO: It seems like these might be lists of strings, this should be checked

        self.term = StringAccessor(target_object, "term")
        self.notes = StringAccessor(target_object, "notes")

    def single_line_desc(self):
        """
            Return a single line string representing the process
        """
        return f"{self.name} {self.date} {self.description}"

    def summary(self):
        return (f"Process:\n"
                f"    Name: {self.name.value}\n"
                f"    Date: {self.date.value}\n"
                f"    Description: {self.description.value}\n"
                f"    Term: {self.term.value}\n"
                f"    Notes: {self.notes.value}\n"
                )

    @staticmethod
    def deserialise_json(json_data: dict):
        name = None
        date = None
        description = None
        term = None
        if "name" in json_data:
            name = json_data["name"]
        if "date" in json_data:
            date = json_data["date"]
        if "description" in json_data:
            description = json_data["description"]
        if "term" in json_data:
            term = json_data["term"]
        return Process(name=name, date=date, description=description, term=term)

    def serialise_json(self):
        return {
            "name": self.name,
            "date": self.date,
            "description": self.description,
            "term": self.term,
        }


@dataclass
class Instrument:
    collimations : list[Collimation]
    source : Source
    detector : list[Detector]

    def summary(self):
        return (
            self.aperture.summary() +
            "\n".join([c.summary for c in self.collimations]) +
            self.detector.summary() +
            self.source.summary())

    @staticmethod
    def deserialize_json(json_data: dict):
        collimations = []
        source = None
        detector= []
        if "collimations" in json_data:
            collimations = [Collimation.deserialise_json(c) for c in json_data["collimations"]]
        if "source" in json_data:
            source = Source.deserialise_json(json_data["source"])
        if "detector" in json_data:
            detector = [Detector.deserialise_json(d) for d in json_data["detector"]]
        return Instrument(collimations=collimations, source=source, detector=detector)

    def serialise_json(self):
        data = {
            "collimations": [c.serialise_json() for c in self.collimations],
            "source": None,
            "detector": [d.serialise_json() for d in self.detector]
        }
        if self.source is not None:
            data["source"] = self.source.serialise_json()
        return data

@dataclass(kw_only=True)
class Metadata:
    title: Optional[str]
    run: list[str]
    definition: str | None
    process: list[Process]
    sample: Sample | None
    instrument: Instrument | None
    raw: MetaNode | None

    def __init__(self, target: AccessorTarget, instrument: Instrument):
        self._target = target

        self.instrument = instrument
        self.process = Process(target.with_path_prefix("sasprocess|process"))
        self.sample = Sample(target.with_path_prefix("sassample|sample"))
        self.transmission_spectrum = TransmissionSpectrum(target.with_path_prefix("sastransmission_spectrum|transmission_spectrum"))

        self._title = StringAccessor(target, "title")
        self._run = StringAccessor(target, "run")
        self._definition = StringAccessor(target, "definition")

        self.title: str = decode_string(self._title.value)
        self.run: str = decode_string(self._run.value)
        self.definition: str = decode_string(self._definition.value)


    def summary(self):
        return (
            f"  {self.title}, Run: {self.run}\n" +
            "  " + "="*len(self.title) +
                           "=======" +
            "="*len(self.run) + "\n\n" +
            f"Definition: {self.title}\n" +
            self.process.summary() +
            self.sample.summary() +
            (self.instrument.summary() if self.instrument else ""))

    @staticmethod
    def deserialize_json(json_data: dict):
        title = json_data["title"]
        run = json_data["run"]
        definition = json_data["definition"]
        process = [Process.deserialise_json(p) for p in json_data["process"]]
        sample = None
        instrument = None
        if json_data["sample"] is not None:
            sample = Sample.deserialise_json(json_data["sample"])
        if json_data["instrument"] is not None:
            instrument = Instrument.deserialize_json(json_data["instrument"])
        return Metadata(
            title=title, run=run, definition=definition, process=process, sample=sample, instrument=instrument
        )

    def serialise_json(self):
        serialized = {
            "instrument": None,
            "process": [p.serialise_json() for p in self.process],
            "sample": None,
            "title": self.title,
            "run": self.run,
            "definition": self.definition
        }
        if self.sample is not None:
            serialized["sample"] = self.sample.serialise_json()
        if self.instrument is not None:
            serialized["instrument"] = self.instrument.serialise_json()

        return serialized

    @property
    def id_header(self):
        """Generate a header for used in the unique_id for datasets"""
        title = ""
        if self.title is not None:
            title = self.title
        return f"{title}:{",".join(self.run)}"

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
            case bytes():
                return obj.decode("utf-8")
            case NamedUnit():
                return obj.name
            case Quantity():
                return {"value": obj.value, "units": obj.units.ascii_symbol}
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
