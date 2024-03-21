# TODO: typing
# TODO: Py2 -> Py3
# TODO: Doc strings
# TODO: Patch so non-breaking


class Vector(object):
    """
    Vector class to hold multi-dimensional objects
    """
    # x component
    x = None
    # y component
    y = None
    # z component
    z = None

    def __init__(self, x=None, y=None, z=None):
        """
        Initialization. Components that are not
        set a set to None by default.

        :param x: x component
        :param y: y component
        :param z: z component
        """
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        msg = "x = %s\ty = %s\tz = %s" % (str(self.x), str(self.y), str(self.z))
        return msg


class Detector(object):
    """
    Class to hold detector information
    """
    # Name of the instrument [string]
    name = None
    # Sample to detector distance [float] [mm]
    distance = None
    distance_unit = 'mm'
    # Offset of this detector position in X, Y,
    # (and Z if necessary) [Vector] [mm]
    offset = None
    offset_unit = 'm'
    # Orientation (rotation) of this detector in roll,
    # pitch, and yaw [Vector] [degrees]
    orientation = None
    orientation_unit = 'degree'
    # Center of the beam on the detector in X and Y
    # (and Z if necessary) [Vector] [mm]
    beam_center = None
    beam_center_unit = 'mm'
    # Pixel size in X, Y, (and Z if necessary) [Vector] [mm]
    pixel_size = None
    pixel_size_unit = 'mm'
    # Slit length of the instrument for this detector.[float] [mm]
    slit_length = None
    slit_length_unit = 'mm'

    def __init__(self):
        """
        Initialize class attribute that are objects...
        """
        self.offset = Vector()
        self.orientation = Vector()
        self.beam_center = Vector()
        self.pixel_size = Vector()

    def __str__(self):
        _str = "Detector:\n"
        _str += "   Name:         %s\n" % self.name
        _str += "   Distance:     %s [%s]\n" % \
            (str(self.distance), str(self.distance_unit))
        _str += "   Offset:       %s [%s]\n" % \
            (str(self.offset), str(self.offset_unit))
        _str += "   Orientation:  %s [%s]\n" % \
            (str(self.orientation), str(self.orientation_unit))
        _str += "   Beam center:  %s [%s]\n" % \
            (str(self.beam_center), str(self.beam_center_unit))
        _str += "   Pixel size:   %s [%s]\n" % \
            (str(self.pixel_size), str(self.pixel_size_unit))
        _str += "   Slit length:  %s [%s]\n" % \
            (str(self.slit_length), str(self.slit_length_unit))
        return _str


class Aperture(object):
    # Name
    name = None
    # Type
    type = None
    # Size name
    size_name = None
    # Aperture size [Vector]
    size = None
    size_unit = 'mm'
    # Aperture distance [float]
    distance = None
    distance_unit = 'mm'

    def __init__(self):
        self.size = Vector()


class Collimation(object):
    """
    Class to hold collimation information
    """
    # Name
    name = None
    # Length [float] [mm]
    length = None
    length_unit = 'mm'
    # Aperture
    aperture = None

    def __init__(self):
        self.aperture = []

    def __str__(self):
        _str = "Collimation:\n"
        _str += "   Length:       %s [%s]\n" % \
            (str(self.length), str(self.length_unit))
        for item in self.aperture:
            _str += "   Aperture size:%s [%s]\n" % \
                (str(item.size), str(item.size_unit))
            _str += "   Aperture_dist:%s [%s]\n" % \
                (str(item.distance), str(item.distance_unit))
        return _str


class Source(object):
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
        self.beam_size = Vector()

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


class Sample(object):
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
        self.position = Vector()
        self.orientation = Vector()
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


class Process(object):
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


