import numpy as np

from sasdata.data import SasData
from sasdata.quantities.constants import TwoPi


class GenericROI:
    """
    Base class used to set up the data from a SasData object for processing.
    This class performs any coordinate system independent setup and validation.
    """

    def __init__(self, center: tuple[float, float] = (0.0, 0.0)):
        """
        Assign the variables used to label the properties of the SasData object.

        In classes inheriting from GenericROI, the variables used to define the
        boundaries of the Region Of Interest are also set up during __init__.
        """
        center_x, center_y = center
        self.center_x = center_x
        self.center_y = center_y
        self.data = None
        self.err_data = None
        self.q_data = None
        self.qx_data = None
        self.qy_data = None

    def validate_and_assign_data(self, data2d: SasData = None) -> None:
        """
        Check that the data supplied is valid and assign data to variables.
        This method must be executed before any further data processing happens.

        :param data2d: A SasData object which is the target of a child class'
                       data manipulations.
        """
        # Check that the supplied data is valid and usable.
        if not isinstance(data2d, SasData):
            msg = "Data supplied must be of type SasData."
            raise TypeError(msg)
        if not ("Qx" in data2d._data_contents and
                "Qy" in data2d._data_contents and
                "I" in data2d._data_contents):
            msg = "SasData object must contain 'Qx', 'Qy', and 'I' data."
            raise TypeError(msg)
        if len(data2d.metadata.instrument.detector) > 1:
            msg = (f"Invalid number of detectors: {len(data2d.metadata.instrument.detector)}."
                   "Cannot have more than 1 detector.")
            raise ValueError(msg)

        # Only use data which is finite and not masked off
        if data2d.mask is not None:
            valid_data = np.isfinite(data2d._data_contents["I"].value) & data2d.mask
        else:
            valid_data = np.isfinite(data2d._data_contents["I"].value)

        # Assign properties of the SasData object to variables for reference
        # during data processing.
        self.data = data2d._data_contents["I"].value[valid_data]
        self.err_data = np.sqrt(data2d._data_contents["I"].variance.value)[valid_data]

        self.qx_data = data2d._data_contents["Qx"].value[valid_data] - self.center_x
        self.qy_data = data2d._data_contents["Qy"].value[valid_data] - self.center_y
        self.q_data = np.sqrt(self.qx_data ** 2 + self.qy_data ** 2)

        # Compute phi in the legacy convention: atan2(qy,qx) + pi
        # (legacy code used this origin; keeping it here makes all polar
        # averaging implementations agree and restores the tests).
        self.phi_data = np.arctan2(self.qy_data, self.qx_data) + np.pi

        # No points should have zero error, if they do then assume the error is
        # the square root of the data. This code was added to replicate
        # previous functionality. It's a bit dodgy, so feel free to remove.
        self.err_data[self.err_data == 0] = np.sqrt(np.abs(self.data[self.err_data == 0]))

class CartesianROI(GenericROI):
    """Base class for data manipulators with a Cartesian (rectangular) ROI."""

    def __init__(self, qx_range: tuple[float, float] = (0.0, 0.0), qy_range: tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Assign the variables used to label the properties of the SasData object.
        Also establish the upper and lower bounds defining the ROI.

        :param qx_range: Bounds of the ROI along the Q_x direction.
        :param qy_range: Bounds of the ROI along the Q_y direction.
        """
        super().__init__()
        qx_min, qx_max = qx_range
        qy_min, qy_max = qy_range
        self.qx_min = qx_min
        self.qx_max = qx_max
        self.qy_min = qy_min
        self.qy_max = qy_max

class PolarROI(GenericROI):
    """Base class for data manipulators with a polar ROI."""

    def __init__(self,
                 r_range: tuple[float, float],
                 phi_range: tuple[float, float] = (0.0, TwoPi),
                 center: tuple[float, float] = (0.0, 0.0)
                 ) -> None:
        """
        Assign the variables used to label the properties of the SasData object.
        Also establish the upper and lower bounds defining the ROI.

        :param r_range: Tuple (r_min, r_max) defining limits for |Q| values to use during averaging.
        :param phi_range: Tuple (phi_min, phi_max) defining limits for φ in the ROI.

        Note that Phi is measured anti-clockwise from the positive x-axis.
        """
        super().__init__(center = center)

        self.phi_data = None

        r_min, r_max = r_range
        phi_min, phi_max = phi_range

        if r_min >= r_max:
            msg = "Minimum radius cannot be greater than maximum radius."
            raise ValueError(msg)

        self.r_min = r_min
        self.r_max = r_max
        self.phi_min = phi_min
        self.phi_max = phi_max
