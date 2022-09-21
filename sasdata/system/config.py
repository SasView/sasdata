class Config:

    def __init__(self):

        # Units to default to when no units found in file
        self.LOADER_I_UNIT_ON_LOAD = "cm^{-1}"
        self.LOADER_Q_UNIT_ON_LOAD = "A^{-1}"

        # Data 1D fields for iterative purposes
        self.FIELDS_1D = ['x', 'y', 'dx', 'dy', 'dxl', 'dxw']
        # Data 2D fields for iterative purposes
        self.FIELDS_2D = ['data', 'qx_data', 'qy_data', 'q_data', 'err_data', 'dqx_data', 'dqy_data', 'mask']

        self.FILE_LOADER_EXTENSIONS = ['.txt']
        self.FILE_LOADER_WLIST = ["Text files (*.txt|*.TXT)"]
        self.EXTENSION_DEPRECATED = ['.asc']


config = Config()
