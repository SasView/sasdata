from tokenize import String

import numpy as np
from numpy.typing import ArrayLike

import sasdata.quantities.units as units
from sasdata.quantities.absolute_temperature import AbsoluteTemperatureAccessor
from sasdata.quantities.accessors import StringAccessor, LengthAccessor, AngleAccessor, QuantityAccessor, \
    DimensionlessAccessor, FloatAccessor, TemperatureAccessor, AccessorTarget


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



class Source:
    """
    Class to hold source information
    """

    def __init__(self, target_object: AccessorTarget):
        # Name
        self.name = StringAccessor(target_object, "name")

        # Generic radiation type (Type and probe give more specific info) [string]
        self.radiation = StringAccessor(target_object, "radiation")

        # Type and probe are only written to by the NXcanSAS reader
        # Specific radiation type (Synchotron X-ray, Reactor neutron, etc) [string]
        self.type = StringAccessor(target_object, "type")

        # Radiation probe (generic probe such as neutron, x-ray, muon, etc) [string]
        self.probe_particle = StringAccessor(target_object, "probe")

        # Beam size name
        self.beam_size_name = StringAccessor(target_object, "beam_size_name")

        # Beam size [Vector] [mm]
        self.beam_size = LengthAccessor[ArrayLike](target_object,
                                                   "beam_size",
                                                   "beam_size.units",
                                                   default_unit=units.millimeters)

        # Beam shape [string]
        self.beam_shape = StringAccessor(target_object, "beam_shape")

        # Wavelength [float] [Angstrom]
        self.wavelength = LengthAccessor[float](target_object,
                                                "wavelength",
                                                "wavelength.units",
                                                default_unit=units.angstroms)

        # Minimum wavelength [float] [Angstrom]
        self.wavelength_min = LengthAccessor[float](target_object,
                                                    "wavelength_min",
                                                    "wavelength_min.units",
                                                    default_unit=units.angstroms)

        # Maximum wavelength [float] [Angstrom]
        self.wavelength_max = LengthAccessor[float](target_object,
                                                    "wavelength_min",
                                                    "wavelength_max.units",
                                                    default_unit=units.angstroms)

        # Wavelength spread [float] [Angstrom]
        # Quantity because it might have other units, such as percent
        self.wavelength_spread = QuantityAccessor[float](target_object,
                                                         "wavelength_spread",
                                                         "wavelength_spread.units",
                                                         default_unit=units.angstroms)

    def summary(self) -> str:

        if self.radiation.value is None and self.type.value and self.probe_particle.value:
            radiation = f"{self.type.value} {self.probe_particle.value}"
        else:
            radiation = f"{self.radiation.value}"

        return (f"Source:\n"
                f"    Radiation:         {radiation}\n"
                f"    Shape:             {self.beam_shape.value}\n"
                f"    Wavelength:        {self.wavelength.value}\n"
                f"    Min. Wavelength:   {self.wavelength_min.value}\n"
                f"    Max. Wavelength:   {self.wavelength_max.value}\n"
                f"    Wavelength Spread: {self.wavelength_spread.value}\n"
                f"    Beam Size:         {self.beam_size.value}\n")



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


class Instrument:
    def __init__(self, target: AccessorTarget, collimations: list[Collimation]):
        self.aperture = Aperture(target.with_path_prefix("sasaperture|aperture"))
        self.collimations = collimations
        self.detector = Detector(target.with_path_prefix("sasdetector|detector"))
        self.source = Source(target.with_path_prefix("sassource|source"))

    def summary(self):
        return (
            self.aperture.summary() +
            "\n".join([c.summary for c in self.collimations]) +
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

class Metadata:
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
            self.instrument.summary() +
            self.transmission_spectrum.summary())
