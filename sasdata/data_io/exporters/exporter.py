
class Exporter:
    # The following are class-level objects that should not be modified at the instance level
    # String to describe the type of data this reader can load
    type_name = ""
    # Wildcards to display
    type = []
    # List of allowed extensions
    ext = []
    # Bypass extension check and try to load anyway
    allow_all = False

    def __init__(self):
        # TODO: Importer and Exporter are similar in concept: Create a top-level Meta class for both
        # A map of Path-like objects to the data that should be exported into that path
        self.data_to_export = {}
        # Path object using the file path sent to reader
        self.filepath = None
        # Starting file position to begin reading data from
        self.f_pos = 0
        # File extension of the data file passed to the reader
        self.extension = None
        # Open file handle
        self.f_open = None

    def write(self):
        pass