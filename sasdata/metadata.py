"""
Contains classes describing the metadata for a scattering run

The metadata is structures around the CANSas format version 1.1, found at
https://www.cansas.org/formats/canSAS1d/1.1/doc/specification.html

Metadata from other file formats should be massaged to fit into the data classes presented here.
Any useful metadata which cannot be included in these classes represent a bug in the CANSas format.

"""

from tokenize import String
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from sasdata.quantities.quantity import Quantity
import sasdata.quantities.units as units
from sasdata.quantities.absolute_temperature import AbsoluteTemperatureAccessor
from sasdata.quantities.accessors import StringAccessor, LengthAccessor, AngleAccessor, QuantityAccessor, \
    DimensionlessAccessor, FloatAccessor, TemperatureAccessor, AccessorTarget

@dataclass(kw_only=True)
class Vec3:
    """A three-vector of measured quantities"""
    x : Quantity[float] | None
    y : Quantity[float] | None
    z : Quantity[float] | None

@dataclass(kw_only=True)
class Rot3:
    """A measured rotation in 3-space"""
    roll : Quantity[float] | None
    pitch : Quantity[float] | None
    yaw : Quantity[float] | None

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

@dataclass(kw_only=True)
class BeamSize:
    name: str | None
    size: Vec3 | None

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
        return f"{self.name.value} {self.date.value} {self.description.value}"

    def summary(self):
        return (f"Process:\n"
                f"    Name: {self.name}\n"
                f"    Date: {self.date}\n"
                f"    Description: {self.description}\n"
                f"    Term: {self.term}\n"
                )


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
