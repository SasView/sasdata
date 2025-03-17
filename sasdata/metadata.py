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

def parse_length(node) -> Quantity[float]:
    """Pull a single quantity with length units out of an HDF5 node"""
    magnitude = node.astype(float)[0]
    unit = node.attrs["units"]
    return Quantity(magnitude, units.symbol_lookup[unit])

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


"""
Definitions of radiation types
"""
NEUTRON = 'neutron'
XRAY = 'x-ray'
MUON = 'muon'
ELECTRON = 'electron'


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
        #
        # _str += "   Details:\n"
        # for item in self.details:
        #     _str += "      %s\n" % item
        #
        # return _str


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
        return f"{self.name.value} {self.date.value} {self.description.value}"

    def summary(self):
        return (f"Process:\n"
                f"    Name: {self.name.value}\n"
                f"    Date: {self.date.value}\n"
                f"    Description: {self.description.value}\n"
                f"    Term: {self.term.value}\n"
                f"    Notes: {self.notes.value}\n"
                )

class TransmissionSpectrum:
    """
    Class that holds information about transmission spectrum
    for white beams and spallation sources.
    """
    def __init__(self, target_object: AccessorTarget):
        # TODO: Needs to be multiple instances
        self.name = StringAccessor(target_object, "name")
        self.timestamp = StringAccessor(target_object, "timestamp")

        # Wavelength (float) [A]
        self.wavelength = LengthAccessor[ArrayLike](target_object,
                                                    "wavelength",
                                                    "wavelength.units")

        # Transmission (float) [unit less]
        self.transmission = DimensionlessAccessor[ArrayLike](target_object,
                                                             "transmission",
                                                             "units",
                                                             default_unit=units.none)

        # Transmission Deviation (float) [unit less]
        self.transmission_deviation = DimensionlessAccessor[ArrayLike](target_object,
                                                                       "transmission_deviation",
                                                                       "transmission_deviation.units",
                                                                       default_unit=units.none)


    def summary(self) -> str:
        return (f"Transmission Spectrum:\n"
                f"    Name:             {self.name.value}\n"
                f"    Timestamp:        {self.timestamp.value}\n"
                f"    Wavelengths:      {self.wavelength.value}\n"
                f"    Transmission:     {self.transmission.value}\n")


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

def decode_string(data):
    """ This is some crazy stuff"""

    if isinstance(data, str):
        return data

    elif isinstance(data, np.ndarray):

        if data.dtype == object:

            data = data.reshape(-1)
            data = data[0]

            if isinstance(data, bytes):
                return data.decode("utf-8")

            return str(data)

        else:
            return data.tobytes().decode("utf-8")

    else:
        return str(data)

class Metadata:
    def __init__(self, target: AccessorTarget, sample: Optional[Sample], instrument: Optional[Instrument]):
        self._target = target

        self.instrument = instrument
        self.process = Process(target.with_path_prefix("sasprocess|process"))
        self.sample = sample
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
            (self.instrument.summary() if self.instrument else "") +
            self.transmission_spectrum.summary())
