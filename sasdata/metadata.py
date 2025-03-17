from tokenize import String
from typing import Optional
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
    x : Optional[Quantity[float]]
    y : Optional[Quantity[float]]
    z : Optional[Quantity[float]]

@dataclass(kw_only=True)
class Rot3:
    """A measured rotation in 3-space"""
    roll : Optional[Quantity[float]]
    pitch : Optional[Quantity[float]]
    yaw : Optional[Quantity[float]]

@dataclass(kw_only=True)
class Detector:
    """
    Detector information
    """
    name : Optional[str]
    distance : Optional[Quantity[float]]
    offset : Optional[Vec3]
    orientation : Optional[Rot3]
    beam_center : Optional[Vec3]
    pixel_size : Optional[Vec3]
    slit_length : Optional[Quantity[float]]


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
    distance: Optional[Quantity[float]]
    size: Optional[Vec3]
    size_name: Optional[str]
    name: Optional[str]
    type_: Optional[str]

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

    length: Quantity[float]
    apertures: list[Aperture]

    def summary(self):

        #TODO collimation stuff
        return (
            f"Collimation:\n"
            f"   Length: {self.length}\n")

@dataclass(kw_only=True)
class BeamSize:
    name: Optional[str]
    size: Optional[Vec3]

@dataclass(kw_only=True)
class Source:
    radiation: str
    beam_shape: Optional[str]
    beam_size: Optional[BeamSize]
    wavelength : Optional[Quantity[float]]
    wavelength_min : Optional[Quantity[float]]
    wavelength_max : Optional[Quantity[float]]
    wavelength_spread : Optional[Quantity[float]]

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
    name: Optional[str]
    sample_id : Optional[str]
    thickness : Optional[Quantity[float]]
    transmission: Optional[float]
    temperature : Optional[Quantity[float]]
    position : Optional[Vec3]
    orientation : Optional[Rot3]
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
    name : Optional[ str ]
    date : Optional[ str ]
    description : Optional[ str ]
    term : Optional[ str ]

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
    source : Source
    detector : list[Detector]

    def summary(self):
        return (
            "\n".join([c.summary() for c in self.collimations]) +
            "".join([d.summary() for d in self.detector]) +
            self.source.summary())

@dataclass(kw_only=True)
class Metadata:
    title: Optional[str]
    run: list[str]
    definition: Optional[str]
    process: list[str]
    sample: Optional[Sample]
    instrument: Optional[Instrument]

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
