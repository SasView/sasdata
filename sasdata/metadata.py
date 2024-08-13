import numpy as np
from numpy.typing import ArrayLike

import sasdata.quantities.units as units
from sasdata.quantities.accessors import StringAccessor, LengthAccessor, AngleAccessor, QuantityAccessor
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
                                                default_units=units.millimeters)

        self.orientation = AngleAccessor[ArrayLike](target_object,
                                                    "detector.orientation",
                                                    "detector.orientation.units",
                                                    default_units=units.degrees)

        self.beam_center = LengthAccessor[ArrayLike](target_object,
                                                     "detector.beam_center",
                                                     "detector.beam_center.units",
                                                     default_units=units.millimeters)

        # Pixel size in X, Y, (and Z if necessary) [Vector] [mm]
        self.pixel_size = LengthAccessor[ArrayLike](target_object,
                                                    "detector.pixel_size",
                                                    "detector.pixel_size.units",
                                                    default_units=units.millimeters)

        # Slit length of the instrument for this detector.[float] [mm]
        self.slit_length = LengthAccessor[float](target_object,
                                                 "detector.slit_length",
                                                 "detector.slit_length.units",
                                                 default_units=units.millimeters)

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
        self.distance = QuantityAccessor[float](self.target_object,
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
        self.length = QuantityAccessor[float](target_object,
                                              "collimation.length",
                                              "collimation.length.units",
                                              default_units=units.millimeters)


        # Todo - how do we handle this
        self.collimator = Collimation(target_object)

    def __str__(self):
        _str = "Collimation:\n"
        _str += "   Length:       %s [%s]\n" % \
            (str(self.length), str(self.length_unit))
        for item in self.aperture:


class Source:
    """
    Class to hold source information
    """
    # Name
    name = None
    # Generic radiation type (Type and probe give more specific info) [string]
    radiation = None
    # Type and probe are only written to by the NXcanSAS reader
    # Specific radiation type (Synchotron X-ray, Reactor neutron, etc) [string]
    type = None
    # Radiation probe (generic probe such as neutron, x-ray, muon, etc) [string]
    probe = None
    # Beam size name
    beam_size_name = None
    # Beam size [Vector] [mm]
    beam_size = None
    beam_size_unit = 'mm'
    # Beam shape [string]
    beam_shape = None
    # Wavelength [float] [Angstrom]
    wavelength = None
    wavelength_unit = 'A'
    # Minimum wavelength [float] [Angstrom]
    wavelength_min = None
    wavelength_min_unit = 'nm'
    # Maximum wavelength [float] [Angstrom]
    wavelength_max = None
    wavelength_max_unit = 'nm'
    # Wavelength spread [float] [Angstrom]
    wavelength_spread = None
    wavelength_spread_unit = 'percent'

    def __init__(self):
        self.beam_size = None #Vector()

    def __str__(self):
        _str = "Source:\n"
        radiation = self.radiation
        if self.radiation is None and self.type and self.probe:
            radiation = self.type + " " + self.probe
        _str += "   Radiation:    %s\n" % str(radiation)
        _str += "   Shape:        %s\n" % str(self.beam_shape)
        _str += "   Wavelength:   %s [%s]\n" % \
            (str(self.wavelength), str(self.wavelength_unit))
        _str += "   Waveln_min:   %s [%s]\n" % \
            (str(self.wavelength_min), str(self.wavelength_min_unit))
        _str += "   Waveln_max:   %s [%s]\n" % \
            (str(self.wavelength_max), str(self.wavelength_max_unit))
        _str += "   Waveln_spread:%s [%s]\n" % \
            (str(self.wavelength_spread), str(self.wavelength_spread_unit))
        _str += "   Beam_size:    %s [%s]\n" % \
            (str(self.beam_size), str(self.beam_size_unit))
        return _str


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
    # Short name for sample
    name = ''
    # ID
    ID = ''
    # Thickness [float] [mm]
    thickness = None
    thickness_unit = 'mm'
    # Transmission [float] [fraction]
    transmission = None
    # Temperature [float] [No Default]
    temperature = None
    temperature_unit = None
    # Position [Vector] [mm]
    position = None
    position_unit = 'mm'
    # Orientation [Vector] [degrees]
    orientation = None
    orientation_unit = 'degree'
    # Details
    details = None
    # SESANS zacceptance
    zacceptance = (0,"")
    yacceptance = (0,"")

    def __init__(self):
        self.position = None # Vector()
        self.orientation = None # Vector()
        self.details = []

    def __str__(self):
        _str = "Sample:\n"
        _str += "   ID:           %s\n" % str(self.ID)
        _str += "   Transmission: %s\n" % str(self.transmission)
        _str += "   Thickness:    %s [%s]\n" % \
            (str(self.thickness), str(self.thickness_unit))
        _str += "   Temperature:  %s [%s]\n" % \
            (str(self.temperature), str(self.temperature_unit))
        _str += "   Position:     %s [%s]\n" % \
            (str(self.position), str(self.position_unit))
        _str += "   Orientation:  %s [%s]\n" % \
            (str(self.orientation), str(self.orientation_unit))

        _str += "   Details:\n"
        for item in self.details:
            _str += "      %s\n" % item

        return _str


class Process:
    """
    Class that holds information about the processes
    performed on the data.
    """
    name = ''
    date = ''
    description = ''
    term = None
    notes = None

    def __init__(self):
        self.term = []
        self.notes = []

    def is_empty(self):
        """
            Return True if the object is empty
        """
        return (len(self.name) == 0 and len(self.date) == 0
                and len(self.description) == 0 and len(self.term) == 0
                and len(self.notes) == 0)

    def single_line_desc(self):
        """
            Return a single line string representing the process
        """
        return "%s %s %s" % (self.name, self.date, self.description)

    def __str__(self):
        _str = "Process:\n"
        _str += "   Name:         %s\n" % self.name
        _str += "   Date:         %s\n" % self.date
        _str += "   Description:  %s\n" % self.description
        for item in self.term:
            _str += "   Term:         %s\n" % item
        for item in self.notes:
            _str += "   Note:         %s\n" % item
        return _str


class TransmissionSpectrum(object):
    """
    Class that holds information about transmission spectrum
    for white beams and spallation sources.
    """
    name = ''
    timestamp = ''
    # Wavelength (float) [A]
    wavelength = None
    wavelength_unit = 'A'
    # Transmission (float) [unit less]
    transmission = None
    transmission_unit = ''
    # Transmission Deviation (float) [unit less]
    transmission_deviation = None
    transmission_deviation_unit = ''

    def __init__(self):
        self.wavelength = []
        self.transmission = []
        self.transmission_deviation = []

    def __str__(self):
        _str = "Transmission Spectrum:\n"
        _str += "   Name:             \t{0}\n".format(self.name)
        _str += "   Timestamp:        \t{0}\n".format(self.timestamp)
        _str += "   Wavelength unit:  \t{0}\n".format(self.wavelength_unit)
        _str += "   Transmission unit:\t{0}\n".format(self.transmission_unit)
        _str += "   Trans. Dev. unit:  \t{0}\n".format(
                                            self.transmission_deviation_unit)
        length_list = [len(self.wavelength), len(self.transmission),
                       len(self.transmission_deviation)]
        _str += "   Number of Pts:    \t{0}\n".format(max(length_list))
        return _str

