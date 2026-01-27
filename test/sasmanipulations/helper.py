"""
Shared test helpers for averaging tests.
"""
import numpy as np
from scipy import integrate

from sasdata.dataloader import data_info
from sasdata.quantities.constants import TwoPi


def make_dd_from_func(func, matrix_size=201):
    """
    Create a MatrixToData2D from a function of (x, y). Returns the MatrixToData2D
    instance and matrix_size for convenience.
    func should accept (x, y) meshgrid arrays and return a 2D array.
    """
    x, y = np.meshgrid(np.linspace(-1, 1, matrix_size),
                       np.linspace(-1, 1, matrix_size))
    mat = func(x, y)
    return MatrixToData2D(data2d=mat), matrix_size

def make_uniform_dd(shape=(100, 100), value=1.0):
    mat = np.full(shape, value, dtype=float)
    return MatrixToData2D(data2d=mat)

def integrate_1d_output(output, method="simpson"):
    """
    Integrate output from an averager consistently.
    - If output is a Data1D-like object with .x and .y -> integrate y(x)
    - If output is a tuple (result, error[, npoints]) -> return numeric result
    """
    if hasattr(output, "x") and hasattr(output, "y"):
        if method == "trapezoid":
            return integrate.trapezoid(output.y, output.x)
        return integrate.simpson(output.y, output.x)
    if isinstance(output, tuple) and len(output) >= 1:
        return output[0]
    raise TypeError("Unsupported averager output type: %r" % type(output))


def expected_slabx_area(qx_min, qx_max, qy_min, qy_max):
    # data = x^2 * y -> integrate x^2 dx and average y across qy range
    x_part_integ = (qx_max**3 - qx_min**3) / 3
    y_part_integ = (qy_max**2 - qy_min**2) / 2
    y_part_avg = y_part_integ / (qy_max - qy_min)
    return y_part_avg * x_part_integ

def expected_slaby_area(qx_min, qx_max, qy_min, qy_max):
    # data = x * y^2 -> integrate y^2 dy and average x across qx range
    y_part_integ = (qy_max**3 - qy_min**3) / 3
    x_part_integ = (qx_max**2 - qx_min**2) / 2
    x_part_avg = x_part_integ / (qx_max - qx_min)
    return x_part_avg * y_part_integ

def make_uniform_dd(shape=(100, 100), value=1.0):
    """Convenience for tests that need a constant matrix Data2D."""
    mat = np.full(shape, value, dtype=float)
    return MatrixToData2D(data2d=mat)

def run_and_integrate(averager, dd, integrator="simpson"):
    """
    Run an averager (callable) with a Data2D container returned by MatrixToData2D
    and return the integrated result (scalar area / sum) consistently.
    """
    out = averager(dd.data)
    return integrate_1d_output(out, method=("trapezoid" if integrator == "trapezoid" else "simpson"))

def expected_boxsum_and_err(matrix, slice_rows=None, slice_cols=None):
    """
    Compute expected Boxsum (sum) and its error for a given 2D numpy matrix.
    Optional slice indices can restrict the region (tuples/lists of indices).
    """
    mat = np.asarray(matrix)
    if slice_rows is not None and slice_cols is not None:
        mat = mat[np.ix_(slice_rows, slice_cols)]
    total = np.sum(mat)
    err = np.sqrt(np.sum(mat))
    return total, err

def expected_boxavg_and_err(matrix, slice_rows=None, slice_cols=None):
    """
    Compute expected Boxavg (mean) and its error for a given 2D numpy matrix.
    Error uses sqrt(sum)/N as in existing tests.
    """
    mat = np.asarray(matrix)
    if slice_rows is not None and slice_cols is not None:
        mat = mat[np.ix_(slice_rows, slice_cols)]
    avg = np.mean(mat) if mat.size > 0 else 0.0
    err = np.sqrt(np.sum(mat)) / mat.size if mat.size > 0 else 0.0
    return avg, err


class MatrixToData2D:
    """
    Create Data2D objects from supplied 2D arrays of data.
    Error data can also be included.

    Adapted from sasdata.data_util.manipulations.reader_2D_converter
    """

    def __init__(self, data2d=None, err_data=None):
        matrix, err_arr = self._validate_and_convert_inputs(data2d, err_data)
        qx_bins, qy_bins = self._compute_bins(matrix.shape)
        data_flat, err_flat, qx_data, qy_data, q_data, mask = self._build_flat_arrays(matrix, err_arr, qx_bins, qy_bins)

        # qmax can be any number, 1 just makes things simple.
        self.qmax = 1
        # Creating a Data2D object to use for testing the averagers.
        self.data = data_info.Data2D(data=data_flat, err_data=err_flat,
                                        qx_data=qx_data, qy_data=qy_data,
                                        q_data=q_data, mask=mask)

    def _validate_and_convert_inputs(self, data2d, err_data):
        """Validate inputs and coerce to numpy arrays. Returns (matrix, err_data_or_None)."""
        if data2d is None:
            raise ValueError("Data must be supplied to convert to Data2D")
        matrix = np.asarray(data2d)
        if matrix.ndim != 2:
            raise ValueError("Supplied array must have 2 dimensions to convert to Data2D")

        if err_data is not None:
            err_arr = np.asarray(err_data)
            if err_arr.shape != matrix.shape:
                raise ValueError("Data and errors must have the same shape")
        else:
            err_arr = None
        return matrix, err_arr

    def _compute_bins(self, matrix_shape):
        """Compute qx and qy bin edges given the matrix shape."""
        cols = matrix_shape[1]
        rows = matrix_shape[0]
        qx_bins = np.linspace(start=-1 * 1, stop=1, num=cols, endpoint=True)
        qy_bins = np.linspace(start=-1 * 1, stop=1, num=rows, endpoint=True)
        return qx_bins, qy_bins
        # qmax can be any number, 1 just makes things simple.
        self.qmax = 1
        qx_bins = np.linspace(start=-1 * self.qmax,
                              stop=self.qmax,
                              num=matrix.shape[1],
                              endpoint=True)
        qy_bins = np.linspace(start=-1 * self.qmax,
                              stop=self.qmax,
                              num=matrix.shape[0],
                              endpoint=True)

    def _build_flat_arrays(self, matrix, err_arr, qx_bins, qy_bins):
        """Flatten matrix and build qx, qy, q arrays plus mask and error handling."""
        data_flat = matrix.flatten()
        if err_arr is None or np.any(err_arr <= 0):
            # Error data of some kind is needed, so we fabricate some
            err_flat = np.sqrt(np.abs(data_flat))
        else:
            err_flat = np.asarray(err_arr).flatten()

        qx_data = np.tile(qx_bins, (len(qy_bins), 1)).flatten()
        qy_data = np.tile(qy_bins, (len(qx_bins), 1)).swapaxes(0, 1).flatten()
        q_data = np.sqrt(qx_data * qx_data + qy_data * qy_data)
        mask = np.ones(len(data_flat), dtype=bool)
        return data_flat, err_flat, qx_data, qy_data, q_data, mask

class CircularTestingMatrix:
    """
    This class is used to generate a 2D array representing a function in polar
    coordinates. The function, f(r, φ) = R(r) * Φ(φ), factorises into simple
    radial and angular parts. This makes it easy to determine the form of the
    function after one of the parts has been averaged over, and therefore good
    for testing the directional averagers in manipulations.py.
    This testing is done by comparing the area under the functions, as these
    will only match if the limits defining the ROI were applied correctly.

    f(r, φ) = R(r) * Φ(φ)
    R(r) = r ; where 0 <= r <= 1.
    Φ(φ) = 1 + sin(ν * φ) ; where ν is the frequency and 0 <= φ <= 2π.
    """

    def __init__(self, frequency=1, matrix_size=201, major_axis=None):
        """
        :param frequency: No. times Φ(φ) oscillates over the 0 <= φ <= 2π range
                          This parameter is largely arbitrary.
        :param matrix_size: The len() of the output matrix.
                            Note that odd numbers give a centrepoint of 0,0.
        :param major_axis: 'Q' or 'Phi' - the axis plotted against by the
                           averager being tested.
        """
        if major_axis not in ('Q', 'Phi'):
            msg = "Major axis must be either 'Q' or 'Phi'."
            raise ValueError(msg)

        self.freq = frequency
        self.matrix_size = matrix_size
        self.major = major_axis

        # Grid with same dimensions as data matrix, ranging from -1 to 1
        x, y = np.meshgrid(np.linspace(-1, 1, self.matrix_size),
                           np.linspace(-1, 1, self.matrix_size))
        # radius is 0 at the centre, and 1 at (0, +/-1) and (+/-1, 0)
        radius = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        # Create the 2D array of data
        # The sinusoidal part is shifted up by 1 so its average is never 0
        self.matrix = radius * (1 + np.sin(self.freq * angle))

    def area_under_region(self, r_min=0, r_max=1, phi_min=0, phi_max=TwoPi):
        """
        Integral of the testing matrix along the major axis, between the limits
        specified. This can be compared to the integral under the 1D data
        output by the averager being tested to confirm it's working properly.
        :param r_min: value defining the minimum Q in the ROI.
        :param r_max: value defining the maximum Q in the ROI.
        :param phi_min: value defining the minimum Phi in the ROI.
        :param phi_max: value defining the maximum Phi in the ROI.
        """

        phi_range = phi_max - phi_min
        # ∫(1 + sin(ν * φ)) dφ = φ + (-cos(ν * φ) / ν) + constant.
        sine_part_integ = phi_range - (np.cos(self.freq * phi_max) -
                                       np.cos(self.freq * phi_min)) / self.freq
        sine_part_avg = sine_part_integ / phi_range

        # ∫(r) dr = r²/2 + constant.
        linear_part_integ = (r_max ** 2 - r_min ** 2) / 2
        # The average radius is weighted towards higher radii. The probability
        # of a point having a given radius value is proportional to the radius:
        # P(r) = k * r ; where k is some proportionality constant.
        # ∫[r₀, r₁] P(r) dr = 1, which can be solved for k. This can then be
        # substituted into ⟨r⟩ = ∫[r₀, r₁] P(r) * r dr, giving:
        linear_part_avg = 2/3 * (r_max**3 - r_min**3) / (r_max**2 - r_min**2)

        # The integral along the major axis is modulated by the average value
        # along the minor axis (between the limits).
        if self.major == 'Q':
            calculated_area = sine_part_avg * linear_part_integ
        else:
            calculated_area = linear_part_avg * sine_part_integ
        return calculated_area
