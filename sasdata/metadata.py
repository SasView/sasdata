"""
Contains classes describing the metadata for a scattering run

The metadata is structures around the CANSas format version 1.1, found at
https://www.cansas.org/formats/canSAS1d/1.1/doc/specification.html

Metadata from other file formats should be massaged to fit into the data classes presented here.
Any useful metadata which cannot be included in these classes represent a bug in the CANSas format.

"""

from dataclasses import dataclass

from sasdata.quantities.quantity import Quantity

@dataclass(kw_only=True)
class Vec3:
    """A three-vector of measured quantities"""
    x : Quantity[float] | None
    y : Quantity[float] | None
    z : Quantity[float] | None

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
    name : str | None
    distance : Quantity[float] | None
    offset : Vec3 | None
    orientation : Rot3 | None
    beam_center : Vec3 | None
    pixel_size : Vec3 | None
    slit_length : Quantity[float] | None


    def summary(self):
        return (f"Detector:\n"
                f"   Name:         {self.name}\n"
                f"   Distance:     {self.distance}\n"
                f"   Offset:       {self.offset}\n"
                f"   Orientation:  {self.orientation}\n"
                f"   Beam center:  {self.beam_center}\n"
                f"   Pixel size:   {self.pixel_size}\n"
                f"   Slit length:  {self.slit_length}\n")

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


@dataclass(kw_only=True)
class Aperture:
    distance: Quantity[float] | None
    size: Vec3 | None
    size_name: str | None
    name: str | None
    type_: str | None

    def summary(self):
        return (f"Aperture:\n"
                f"  Name: {self.name}\n"
                f"  Aperture size: {self.size}\n"
                f"  Aperture distance: {self.distance}\n")

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

@dataclass(kw_only=True)
class Collimation:
    """
    Class to hold collimation information
    """

    length: Quantity[float] | None
    apertures: list[Aperture]

    def summary(self):

        #TODO collimation stuff
        return (
            f"Collimation:\n"
            f"   Length: {self.length}\n")

    def serialise_json(self):
        data = {
            "length": None,
            "apertures": [a.serialise_json() for a in self.apertures]
        }
        if self.length is not None:
            data["length"] = self.length.serialise_json()
        return data

@dataclass(kw_only=True)
class BeamSize:
    name: str | None
    size: Vec3 | None

    def serialise_json(self):
        data = {
            "name": self.name,
            "size": None
        }
        if self.size is not None:
            data["size"] = self.size.serialise_json()
        return data

@dataclass(kw_only=True)
class Source:
    radiation: str | None
    beam_shape: str | None
    beam_size: BeamSize | None
    wavelength : Quantity[float] | None
    wavelength_min : Quantity[float] | None
    wavelength_max : Quantity[float] | None
    wavelength_spread : Quantity[float] | None

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

@dataclass(kw_only=True)
class Sample:
    """
    Class to hold the sample description
    """
    name: str | None
    sample_id : str | None
    thickness : Quantity[float] | None
    transmission: float | None
    temperature : Quantity[float] | None
    position : Vec3 | None
    orientation : Rot3 | None
    details : list[str]

    def summary(self) -> str:
        return (f"Sample:\n"
                f"   ID:           {self.sample_id}\n"
                f"   Transmission: {self.transmission}\n"
                f"   Thickness:    {self.thickness}\n"
                f"   Temperature:  {self.temperature}\n"
                f"   Position:     {self.position}\n"
                f"   Orientation:  {self.orientation}\n")

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


@dataclass(kw_only=True)
class Process:
    """
    Class that holds information about the processes
    performed on the data.
    """
    name :  str  | None
    date :  str  | None
    description :  str  | None
    term :  str  | None

    def single_line_desc(self):
        """
            Return a single line string representing the process
        """
        return f"{self.name} {self.date} {self.description}"

    def summary(self):
        return (f"Process:\n"
                f"    Name: {self.name}\n"
                f"    Date: {self.date}\n"
                f"    Description: {self.description}\n"
                f"    Term: {self.term}\n"
                )

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
    source : Source | None
    detector : list[Detector]

    def summary(self):
        return (
            "\n".join([c.summary() for c in self.collimations]) +
            "".join([d.summary() for d in self.detector]) +
            self.source.summary())

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
    title: str | None
    run: list[str]
    definition: str | None
    process: list[Process]
    sample: Sample | None
    instrument: Instrument | None

    def summary(self):
        run_string = self.run[0] if len(self.run) == 1 else self.run
        return (
            f"  {self.title}, Run: {run_string}\n" +
            "  " + "="*len(self.title) +
                           "=======" +
            "="*len(run_string) + "\n\n" +
            f"Definition: {self.title}\n" +
            "".join([p.summary() for p in self.process]) +
            self.sample.summary() +
            (self.instrument.summary() if self.instrument else ""))

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