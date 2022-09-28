"""
    Module that contains classes to hold information read from
    reduced data files.

    A good description of the data members can be found in
    the CanSAS 1D XML data format:

    http://www.smallangles.net/wgwiki/index.php/cansas1d_documentation
"""

from sasdata.data.meta_data import Sample, Source, Process


class DataInfo:
    """
    Class to hold the data read from a file.
    It includes four blocks of data for the
    instrument description, the sample description,
    the data itself and any other meta data.
    """
    # Title
    title = ''
    # Run number
    run = None
    # Run name
    run_name = None
    # File name
    filename = ''
    # Notes
    notes = None
    # Processes (Action on the data)
    process = None
    # Instrument name
    instrument = ''
    # Detector information
    detector = None
    # Sample information
    sample = None
    # Source information
    source = None
    # Collimation information
    collimation = None
    # Transmission Spectrum INfo
    trans_spectrum = None
    # Additional meta-data
    meta_data = None
    # Loading errors
    errors = None
    # SESANS data check
    isSesans = None

    def __init__(self):
        """
        Initialization
        """
        # Title
        self.title = ''
        # Run number
        self.run = []
        self.run_name = {}
        # File name
        self.filename = ''
        # Notes
        self.notes = []
        # Processes (Action on the data)
        self.process = []
        # Instrument name
        self.instrument = ''
        # Detector information
        self.detector = []
        # Sample information
        self.sample = Sample()
        # Source information
        self.source = Source()
        # Collimation information
        self.collimation = []
        # Transmission Spectrum
        self.trans_spectrum = []
        # Additional meta-data
        self.meta_data = {}
        # Loading errors
        self.errors = []
        # SESANS data check
        self.isSesans = False

    def append_empty_process(self):
        """
        """
        self.process.append(Process())

    def add_notes(self, message=""):
        """
        Add notes to datainfo
        """
        self.notes.append(message)

    def __str__(self):
        """
        Nice printout
        """
        _str = f"File:            {self.filename}\n"
        _str += f"Title:           {self.title}\n"
        _str += f"Run:             {self.run}\n"
        _str += f"SESANS:          {self.isSesans}\n"
        _str += f"Instrument:      {self.instrument}\n"
        _str += f"{str(self.sample)}\n"
        _str += f"{str(self.source)}\n"
        for item in self.detector:
            _str += f"{str(item)}\n"
        for item in self.collimation:
            _str += f"{str(item)}\n"
        for item in self.process:
            _str += f"{str(item)}\n"
        for item in self.notes:
            _str += f"{str(item)}\n"
        for item in self.trans_spectrum:
            _str += f"{str(item)}\n"
        return _str

    # Private method to perform operation. Not implemented for DataInfo,
    # but should be implemented for each data class inherited from DataInfo
    # that holds actual data (ex.: Data1D)
    def _perform_operation(self, other, operation):
        """
        Private method to perform operation. Not implemented for DataInfo,
        but should be implemented for each data class inherited from DataInfo
        that holds actual data (ex.: Data1D)
        """
        return NotImplemented

    def _perform_union(self, other):
        """
        Private method to perform union operation. Not implemented for DataInfo,
        but should be implemented for each data class inherited from DataInfo
        that holds actual data (ex.: Data1D)
        """
        return NotImplemented

    def __add__(self, other):
        """
        Add two data sets

        :param other: data set to add to the current one
        :return: new data set
        :raise ValueError: raised when two data sets are incompatible
        """
        def operation(a, b):
            return a + b
        return self._perform_operation(other, operation)

    def __radd__(self, other):
        """
        Add two data sets

        :param other: data set to add to the current one
        :return: new data set
        :raise ValueError: raised when two data sets are incompatible
        """
        def operation(a, b):
            return b + a
        return self._perform_operation(other, operation)

    def __sub__(self, other):
        """
        Subtract two data sets

        :param other: data set to subtract from the current one
        :return: new data set
        :raise ValueError: raised when two data sets are incompatible
        """
        def operation(a, b):
            return a - b
        return self._perform_operation(other, operation)

    def __rsub__(self, other):
        """
        Subtract two data sets

        :param other: data set to subtract from the current one
        :return: new data set
        :raise ValueError: raised when two data sets are incompatible
        """
        def operation(a, b):
            return b - a
        return self._perform_operation(other, operation)

    def __mul__(self, other):
        """
        Multiply two data sets

        :param other: data set to subtract from the current one
        :return: new data set
        :raise ValueError: raised when two data sets are incompatible
        """
        def operation(a, b):
            return a * b
        return self._perform_operation(other, operation)

    def __rmul__(self, other):
        """
        Multiply two data sets

        :param other: data set to subtract from the current one
        :return: new data set
        :raise ValueError: raised when two data sets are incompatible
        """
        def operation(a, b):
            return b * a
        return self._perform_operation(other, operation)

    def __truediv__(self, other):
        """
        Divided a data set by another

        :param other: data set that the current one is divided by
        :return: new data set
        :raise ValueError: raised when two data sets are incompatible
        """
        def operation(a, b):
            return a/b
        return self._perform_operation(other, operation)
    __div__ = __truediv__

    def __rtruediv__(self, other):
        """
        Divided a data set by another

        :param other: data set that the current one is divided by
        :return: new data set
        :raise ValueError: raised when two data sets are incompatible
        """
        def operation(a, b):
            return b/a
        return self._perform_operation(other, operation)
    __rdiv__ = __rtruediv__

    def __or__(self, other):
        """
        Union a data set with another

        :param other: data set to be unified
        :return: new data set
        :raise ValueError: raised when two data sets are incompatible
        """
        return self._perform_union(other)

    def __ror__(self, other):
        """
        Union a data set with another

        :param other: data set to be unified
        :return: new data set
        :raise ValueError: raised when two data sets are incompatible
        """
        return self._perform_union(other)


