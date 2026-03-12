
import numpy as np
from numpy.typing import ArrayLike

import sasdata.quantities.units as units
from sasdata.quantities.absolute_temperature import AbsoluteTemperatureAccessor
from sasdata.quantities.accessors import (
    AccessorTarget,
    AngleAccessor,
    FloatAccessor,
    LengthAccessor,
    QuantityAccessor,
    StringAccessor,
)


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
                f"   Name:         {self.name.value}\n"
                f"   Distance:     {self.distance.value}\n"
                f"   Offset:       {self.offset.value}\n"
                f"   Orientation:  {self.orientation.value}\n"
                f"   Beam center:  {self.beam_center.value}\n"
                f"   Pixel size:   {self.pixel_size.value}\n"
                f"   Slit length:  {self.slit_length.value}\n")

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
                f"  Name: {self.name.value}\n"
                f"  Aperture size: {self.size.value}\n"
                f"  Aperture distance: {self.distance.value}\n")

class Collimation:
    """
    Class to hold collimation information
    """

    def __init__(self, target_object: AccessorTarget):

        # Name
        self.name = StringAccessor(target_object, "name")
        # Length [float] [mm]
        self.length = LengthAccessor[float](target_object,
                                              "length",
                                              "length.units",
                                              default_unit=units.millimeters)


        # Todo - how do we handle this
        # self.collimator = Collimation(target_object)

    def summary(self):

        #TODO collimation stuff
        return (
            f"Collimation:\n"
            f"   Length: {self.length.value}\n")

@dataclass
class BeamSize:
    name: Optional[str]
    x: Optional[Quantity[float]]
    y: Optional[Quantity[float]]
    z: Optional[Quantity[float]]


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
                f"   ID:           {self.sample_id.value}\n"
                f"   Transmission: {self.transmission.value}\n"
                f"   Thickness:    {self.thickness.value}\n"
                f"   Temperature:  {self.temperature.value}\n"
                f"   Position:     {self.position.value}\n"
                f"   Orientation:  {self.orientation.value}\n")
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


@dataclass
class Instrument:
    collimations : list[Collimation]
    source : Source
    detector : list[Detector]

    def summary(self):
        return (
            self.aperture.summary() +
            self.collimation.summary() +
            self.detector.summary() +
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

@dataclass(kw_only=True)
class Metadata:
    def __init__(self, target: AccessorTarget):
        self._target = target

        self.instrument = Instrument(target.with_path_prefix("sasinstrument|instrument"))
        self.process = Process(target.with_path_prefix("sasprocess|process"))
        self.sample = Sample(target.with_path_prefix("sassample|sample"))
        self.transmission_spectrum = TransmissionSpectrum(target.with_path_prefix("sastransmission_spectrum|transmission_spectrum"))

        self._title = StringAccessor(target, "title")
        self._run = StringAccessor(target, "run")
        self._definition = StringAccessor(target, "definition")

        self.title: str = decode_string(self._title.value)
        self.run: str = decode_string(self._run.value)
        self.definition: str = decode_string(self._definition.value)
    title: Optional[str]
    run: list[str]
    definition: Optional[str]
    process: list[str]
    sample: Optional[Sample]
    instrument: Optional[Instrument]

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
