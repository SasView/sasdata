import numpy as np
from typing import Optional, Iterable

from sasdata.data_util.deprecation import deprecated


class Plottable:
    """Base class all plottable objects should inherit from."""

    # Data
    _x: Optional[Iterable] = None
    _y: Optional[Iterable] = None
    _dx: Optional[Iterable] = None
    _dy: Optional[Iterable] = None

    # Units
    _x_unit: str = ''
    _y_unit: str = ''

    # Plot Axis Titles
    _x_label: str = ''
    _y_label: str = ''

    # Min/Max
    _x_min: Optional[float] = None
    _x_max: Optional[float] = None
    _y_min: Optional[float] = None
    _y_max: Optional[float] = None

    # Plot properties
    _mask: Optional[Iterable] = None

    # Flags
    _is_sesans: bool = False

    def __init__(self, x: Iterable, y: Iterable,
                 dx: Optional[Iterable] = None, dy: Optional[Iterable] = None, mask: Optional[Iterable] = None):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.mask = mask

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x: Optional[Iterable]):
        self._x = np.asarray(x) if x is not None else None

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y: Optional[Iterable]):
        self._y = np.asarray(y) if y is not None else None

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, dx: Optional[Iterable]):
        self._dx = np.asarray(dx) if dx is not None else None

    @property
    def dy(self):
        return self._dy

    @dy.setter
    def dy(self, dy: Optional[Iterable]):
        self._dy = np.asarray(dy) if dy is not None else None

    @property
    def x_unit(self):
        return self._x_unit

    @x_unit.setter
    def x_unit(self, unit: str):
        # TODO: sanitize the inputs
        self._x_unit = unit

    @property
    def y_unit(self):
        return self._y_unit

    @y_unit.setter
    def y_unit(self, unit: str):
        # TODO: sanitize the inputs
        self._y_unit = unit

    @property
    def x_label(self):
        return self._x_label

    @x_label.setter
    def x_label(self, title: str):
        # TODO: Sanitize title
        self._x_label = title

    @property
    def y_label(self):
        return self._y_label

    @y_label.setter
    def y_label(self, title: str):
        # TODO: Sanitize title
        self._y_label = title

    @property
    def x_max(self):
        self._x_max = max(self.x) if any(self.x) else None
        return self._x_max

    @property
    def x_min(self):
        self._x_min = min(self.x) if any(self.x) else None
        return self._x_min

    @property
    def y_max(self):
        self._y_max = max(self.y) if any(self.y) else None
        return self._y_max

    @property
    def y_min(self):
        self._y_min = min(self.y) if any(self.y) else None
        return self._y_min

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask: Optional[Iterable]):
        self._mask = np.asarray(mask) if mask is not None else None

    def x_axis(self, label: str, unit: str):
        self.x_label = label
        self.x_unit = unit

    def y_axis(self, label: str, unit: str):
        self.y_label = label
        self.y_unit = unit

    #################
    # Deprecated properties below here

    @deprecated(replaced_with='self.x_max')
    @property
    def xmax(self):
        return self.x_max

    @deprecated(replaced_with='self.x_max')
    @property
    def xmin(self):
        return self.x_min

    @deprecated(replaced_with='self.y_max')
    @property
    def ymax(self):
        return self.y_max

    @deprecated(replaced_with='self.y_min')
    @property
    def ymin(self):
        return self.y_min

    @deprecated(replaced_with='self.x_axis')
    def xaxis(self, label: str, unit: str):
        self.x_axis(label, unit)

    @deprecated(replaced_with='self.y_axis')
    def yaxis(self, label: str, unit: str):
        self.y_label = label
        self.y_unit = unit

    @deprecated(replaced_with='SpinEchoSANS class')
    def isSesans(self):
        return self._is_sesans

    @isSesans.setter
    def isSesans(self, is_sesans_data: bool):
        self._is_sesans = is_sesans_data

    # TODO: Add, subtract, multiple divide abstract methods here? -> manipulations performed...
    # TODO: Unit conversion (for ALL data objects)


class Plottable1D(Plottable):
    """Data class for generic 1-dimensional data. This will typically be SAS data in the form I vs. Q."""

    def __init__(self, x: Iterable, y: Iterable,
                 dx: Optional[Iterable] = None, dy: Optional[Iterable] = None, mask: Optional[Iterable] = None):
        super().__init__(x, y, dx, dy, mask)


class SlitSmeared1D(Plottable):
    """Data class for slit-smeared 1-dimensional data. This will typically be SAS data in the form I vs. Q."""

    # Slit smeared resolution
    _dxl = None
    _dxw = None

    def __init__(self, x: Iterable, y: Iterable,
                 dx: Optional[Iterable] = None, dy: Optional[Iterable] = None, mask: Optional[Iterable] = None,
                 dxl: Optional[Iterable] = None, dxw: Optional[Iterable] = None):
        super().__init__(x, y, dx, dy, mask)
        self.dxl = dxl
        self.dxw = dxw

    @property
    def dxl(self):
        return self._dxl

    @dxl.setter
    def dxl(self, dxl: Optional[Iterable]):
        self._dxl = np.asarray(dxl) if dxl is not None else None

    @property
    def dxw(self):
        return self._dxw

    @dxw.setter
    def dxw(self, dxw: Optional[Iterable]):
        self._dxw = np.asarray(dxw) if dxw is not None else None


class SpinEchoSANS(Plottable):
    """Data class for SESANS data."""

    _lam = None
    _dlam = None

    # TODO: Make into property and add deprecation
    isSesans = True

    def __init__(self, x: Iterable, y: Iterable,
                 dx: Optional[Iterable] = None, dy: Optional[Iterable] = None,
                 mask: Optional[Iterable] = None,
                 lam: Optional[Iterable] = None, dlam: Optional[Iterable] = None):
        super().__init__(x, y, dx, dy, mask)
        self.lam = lam
        self.dlam = dlam

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, lam: Optional[Iterable]):
        self._lam = np.asarray(lam) if lam is not None else None

    @property
    def dlam(self):
        return self._dlam

    @dlam.setter
    def dlam(self, dlam: Optional[Iterable]):
        self._dlam = np.asarray(dlam) if dlam is not None else None


class Plottable2D(Plottable):
    """Data class for generic 2-dimensional data. This will typically be SAS data in the I(Qx, Qy) format."""

    # Data
    _z = None
    _dz = None

    # Units
    _z_unit = ''

    # Plot Axis Titles
    _z_label = ''

    # Min/Max
    _z_min = None
    _z_max = None

    # Qx and Qy bins
    _x_bins = None
    _y_bins = None

    ##################################################
    #
    # Deprecated properties that will be removed in a future release
    @deprecated(replaced_with='Plottable2D.x')
    @property
    def qx_data(self):
        return self.x

    @qx_data.setter
    def qx_data(self, x: Iterable):
        self.x = x

    @deprecated(replaced_with='Plottable2D.y')
    @property
    def qy_data(self):
        return self.y

    @qy_data.setter
    def qy_data(self, y: Iterable):
        self.y = y

    @deprecated(replaced_with='Plottable2D.z')
    @property
    def data(self):
        return self.z

    @data.setter
    def data(self, z: Iterable):
        self.z = z

    @deprecated(replaced_with='Plottable2D.dx')
    @property
    def dqx_data(self):
        return self.dx

    @dqx_data.setter
    def dqx_data(self, dx: Iterable):
        self.dx = dx

    @deprecated(replaced_with='Plottable2D.y')
    @property
    def dqy_data(self):
        return self.dy

    @dqy_data.setter
    def dqy_data(self, dy: Iterable):
        self.dy = dy

    @deprecated(replaced_with='Plottable2D.z')
    @property
    def error_data(self):
        return self.dz

    @error_data.setter
    def error_data(self, dz: Iterable):
        self.dz = dz
    # End of deprecated properties
    #
    ##################################################

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, z: Iterable):
        self._z = z

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, dz: Optional[Iterable]):
        self._dz = np.asarray(dz) if dz is not None else None

    @property
    def z_unit(self):
        return self._z_unit

    @z_unit.setter
    def z_unit(self, unit: str):
        # TODO: sanitize the inputs
        self._z_unit = unit

    @property
    def z_max(self):
        self._z_max = max(self.z) if any(self.z) else None
        return self._z_max

    @property
    def z_min(self):
        self._z_min = min(self.z) if any(self.z) else None
        return self._z_min

    @deprecated(replaced_with='self.z_min')
    @property
    def zmin(self):
        return self.z_min

    @property
    def x_bins(self):
        return self._x_bins

    @x_bins.setter
    def x_bins(self, bins: Optional[Iterable]):
        self._x_bins = bins

    @property
    def y_bins(self):
        return self._y_bins

    @y_bins.setter
    def y_bins(self, bins: Optional[Iterable]):
        self._y_bins = bins

    # TODO: zaxis

    def __init__(self, x: Iterable, y: Iterable, z: Iterable,
                 dx: Optional[Iterable] = None, dy: Optional[Iterable] = None, dz: Optional[Iterable] = None,
                 mask: Optional[Iterable] = None):
        super().__init__(x, y, dx, dy, mask)
        self.z = z
        self.dz = dz
        # TODO: populate min/max and bins


# TODO: Add a 2D slit smeared data object
# TODO: define what resolution should be used in some meaningful way (remove conditionals in sasmodels.direct_model)
# TODO: different data types (refl vs sans vs saxs vs dls, etc)
# TODO: different resolution functions (uniform, vs. empirical, vs. gaussian, etc)
# TODO: Add empty data set generation - pull from sasmodels.data
# TODO: AbstractFittingEngine - replace data class with what is here (what else is needed?)
