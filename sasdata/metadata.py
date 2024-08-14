import numpy as np
from numpy.typing import ArrayLike

import sasdata.quantities.units as units
from quantities.absolute_temperature import AbsoluteTemperatureAccessor
from sasdata.quantities.accessors import StringAccessor, LengthAccessor, AngleAccessor, QuantityAccessor, \
    DimensionlessAccessor, FloatAccessor, TemperatureAccessor


class Detector:
    """
    Detector information
    """

    def __init__(self, target_object):

        # Name of the instrument [string]
        self.name = StringAccessor(target_object, "detector.name")

        # Sample to detector distance [float] [mm]
        self.distance = LengthAccessor[float](target_object,
                                              "detector.distance",
                                              "detector.distance.units",
                                              default_unit=units.millimeters)

        # Offset of this detector position in X, Y,
        # (and Z if necessary) [Vector] [mm]
        self.offset = LengthAccessor[ArrayLike](target_object,
                                                "detector.offset",
                                                "detector.offset.units",
                                                default_unit=units.millimeters)

        self.orientation = AngleAccessor[ArrayLike](target_object,
                                                    "detector.orientation",
                                                    "detector.orientation.units",
                                                    default_unit=units.degrees)

        self.beam_center = LengthAccessor[ArrayLike](target_object,
                                                     "detector.beam_center",
                                                     "detector.beam_center.units",
                                                     default_unit=units.millimeters)

        # Pixel size in X, Y, (and Z if necessary) [Vector] [mm]
        self.pixel_size = LengthAccessor[ArrayLike](target_object,
                                                    "detector.pixel_size",
                                                    "detector.pixel_size.units",
                                                    default_unit=units.millimeters)

        # Slit length of the instrument for this detector.[float] [mm]
        self.slit_length = LengthAccessor[float](target_object,
                                                 "detector.slit_length",
                                                 "detector.slit_length.units",
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

    def __init__(self, target_object):

        # Name
        self.name = StringAccessor(target_object, "aperture.name")

        # Type
        self.type = StringAccessor(target_object, "aperture.type")

        # Size name - TODO: What is the name of a size
        self.size_name = StringAccessor(target_object, "aperture.size_name")

        # Aperture size [Vector] # TODO: Wat!?!
        self.size = QuantityAccessor[ArrayLike](target_object,
                                "aperture.size",
                                "aperture.size.units",
                                default_unit=units.millimeters)

        # Aperture distance [float]
        self.distance = LengthAccessor[float](target_object,
                                    "apature.distance",
                                    "apature.distance.units",
                                    default_unit=units.millimeters)


    def summary(self):
        return (f"Aperture:\n"
                f"  Name: {self.name.value}\n"
                f"  Aperture size: {self.size.value}\n"
                f"  Aperture distance: {self.distance.value}")

class Collimation:
    """
    Class to hold collimation information
    """

    def __init__(self, target_object):

        # Name
        self.name = StringAccessor(target_object, "collimation.name")
        # Length [float] [mm]
        self.length = LengthAccessor[float](target_object,
                                              "collimation.length",
                                              "collimation.length.units",
                                              default_unit=units.millimeters)


        # Todo - how do we handle this
        self.collimator = Collimation(target_object)

    def summary(self):

        #TODO collimation stuff
        return (
            f"Collimation:\n"
            f"   Length: {self.length.value}\n")



class Source:
    """
    Class to hold source information
    """

    def __init__(self, target_object):
        # Name
        self.name = StringAccessor(target_object, "source.name")

        # Generic radiation type (Type and probe give more specific info) [string]
        self.radiation = StringAccessor(target_object, "source.radiation")

        # Type and probe are only written to by the NXcanSAS reader
        # Specific radiation type (Synchotron X-ray, Reactor neutron, etc) [string]
        self.type = StringAccessor(target_object, "source.type")

        # Radiation probe (generic probe such as neutron, x-ray, muon, etc) [string]
        self.probe_particle = StringAccessor(target_object, "source.probe")

        # Beam size name
        self.beam_size_name = StringAccessor(target_object, "source.beam_size_name")

        # Beam size [Vector] [mm]
        self.beam_size = LengthAccessor[ArrayLike](target_object,
                                                   "source.beam_size",
                                                   "source.beam_size.units",
                                                   default_unit=units.millimeters)

        # Beam shape [string]
        self.beam_shape = StringAccessor(target_object, "source.beam_shape")

        # Wavelength [float] [Angstrom]
        self.wavelength = LengthAccessor[float](target_object,
                                                "source.wavelength",
                                                "source.wavelength.units",
                                                default_unit=units.angstroms)

        # Minimum wavelength [float] [Angstrom]
        self.wavelength_min = LengthAccessor[float](target_object,
                                                    "source.wavelength_min",
                                                    "source.wavelength_min.units",
                                                    default_unit=units.angstroms)

        # Maximum wavelength [float] [Angstrom]
        self.wavelength_max = LengthAccessor[float](target_object,
                                                    "source.wavelength_min",
                                                    "source.wavelength_max.units",
                                                    default_unit=units.angstroms)

        # Wavelength spread [float] [Angstrom]
        # Quantity because it might have other units, such as percent
        self.wavelength_spread = QuantityAccessor[float](target_object,
                                                         "source.wavelength_spread",
                                                         "source.wavelength_spread.units",
                                                         default_unit=units.angstroms)

    def summary(self) -> str:

        if self.radiation.value is None and self.type.value and self.probe_particle.value:
            radiation = f"{self.type.value} {self.probe_particle.value}"
        else:
            radiation = f"{self.radiation.value}"

        return (f"Source:\n"
                f"    Radiation: {radiation}\n"
                f"    Shape: {self.beam_shape.value}\n"
                f"    Wavelength: {self.wavelength.value}\n"
                f"    Min. Wavelength: {self.wavelength_min.value}\n"
                f"    Max. Wavelength: {self.wavelength_max.value}\n"
                f"    Wavelength Spread: {self.wavelength_spread.value}\n"
                f"    Beam Size: {self.beam_size}\n")



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
    def __init__(self, target_object):

        # Short name for sample
        self.name = StringAccessor(target_object, "sample.name")
        # ID

        self.sample_id = StringAccessor(target_object, "sample.id")

        # Thickness [float] [mm]
        self.thickness = LengthAccessor(target_object,
                                        "sample.thickness",
                                        "sample.thickness.units",
                                        default_unit=units.millimeters)

        # Transmission [float] [fraction]
        self.transmission = FloatAccessor(target_object,"sample.transmission")

        # Temperature [float] [No Default]
        self.temperature = AbsoluteTemperatureAccessor(target_object,
                                                       "sample.temperature",
                                                       "sample.temperature.unit",
                                                       default_unit=units.kelvin)
        # Position [Vector] [mm]
        self.position = LengthAccessor[ArrayLike](target_object,
                                                  "sample.position",
                                                  "sample.position.unit",
                                                  default_unit=units.millimeters)

        # Orientation [Vector] [degrees]
        self.orientation = AngleAccessor[ArrayLike](target_object,
                                                    "sample.orientation",
                                                    "sample.orientation.unit",
                                                    default_unit=units.degrees)

        # Details
        self.details = StringAccessor(target_object, "sample.details")


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
    def __init__(self, target_object):
        self.name = StringAccessor(target_object, "process.name")
        self.date = StringAccessor(target_object, "process.date")
        self.description = StringAccessor(target_object, "process.description")

        #TODO: It seems like these might be lists of strings, this should be checked

        self.term = StringAccessor(target_object, "process.term")
        self.notes = StringAccessor(target_object, "process.notes")

    def single_line_desc(self):
        """
            Return a single line string representing the process
        """
        return f"{self.name.value} {self.date.value} {self.description.value}"

    def __str__(self):
        return (f"Process:\n"
                f"    Name: {self.name.value}\n"
                f"    Date: {self.date.value}\n"
                f"    Description: {self.description.value}\n"
                f"    Term: {self.term.value}\n"
                f"    Notes: {self.notes.value}"
                )

class TransmissionSpectrum:
    """
    Class that holds information about transmission spectrum
    for white beams and spallation sources.
    """
    def __init__(self, target_object):
        # TODO: Needs to be multiple cases
        self.name = StringAccessor(target_object, "transmission.")
        self.timestamp = StringAccessor(target_object, "transmission.timestamp")

        # Wavelength (float) [A]
        self.wavelength = LengthAccessor[ArrayLike](target_object,
                                                    "transmission.wavelength",
                                                    "transmission.wavelength.units")

        # Transmission (float) [unit less]
        self.transmission = DimensionlessAccessor[ArrayLike](target_object,
                                                             "transmission.transmission",
                                                             "transmission.units",
                                                             default_unit=units.none)

        # Transmission Deviation (float) [unit less]
        self.transmission_deviation = DimensionlessAccessor[ArrayLike](target_object,
                                                                       "transmission.transmission_deviation",
                                                                       "transmission.transmission_deviation.units",
                                                                       default_units=units.none)


    def summary(self) -> str:
        return (f"Transmission Spectrum:\n"
                f"    Name:             {self.name.value}\n"
                f"    Timestamp:        {self.timestamp.value}\n"
                f"    Wavelengths:      {self.wavelength.value}\n"
                f"    Transmission:     {self.transmission.value}\n")

