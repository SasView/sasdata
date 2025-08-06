from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from sasdata.slicing.meshes.mesh import Mesh
from sasdata.slicing.meshes.voronoi_mesh import voronoi_mesh
from sasdata.slicing.meshes.meshmerge import meshmerge

import time

@dataclass
class CacheData:
    """ Data cached for repeated calculations with the same coordinates """
    input_coordinates: np.ndarray  # Input data
    input_coordinates_mesh: Mesh   # Mesh of the input data
    merged_mesh_data: tuple[Mesh, np.ndarray, np.ndarray] # mesh information about the merging


class Rebinner(ABC):


    def __init__(self):
        """ Base class for rebinning methods"""

        self._bin_mesh_cache: Mesh | None = None # cached version of the output bin mesh

        # Output dependent caching
        self._input_cache: CacheData | None = None


    @abstractmethod
    def _bin_coordinates(self) -> np.ndarray:
        """ Coordinates for the output bins """

    @abstractmethod
    def _bin_mesh(self) -> Mesh:
        """ Get the meshes used for binning """

    @property
    def allowable_orders(self) -> list[int]:
        return [-1, 0, 1]

    @property
    def bin_mesh(self) -> Mesh:

        if self._bin_mesh_cache is None:
            bin_mesh = self._bin_mesh()
            self._bin_mesh_cache = bin_mesh

        return self._bin_mesh_cache

    def _post_processing(self, coordinates, values) -> tuple[np.ndarray, np.ndarray]:
        """ Perform post-processing on the mesh binned values """
        # Default is to do nothing, override if needed
        return coordinates, values

    def _calculate(self, input_coordinates: np.ndarray, input_data: np.ndarray, order: int) -> np.ndarray:
        """ Main calculation """

        if order == -1:
            # Construct the input output mapping just based on input points being the output cells,
            # Equivalent to the original binning method

            mesh = self.bin_mesh
            bin_identities = mesh.locate_points(input_coordinates[:,0], input_coordinates[:, 1])
            output_data = np.zeros(mesh.n_cells, dtype=float)

            for index, bin in enumerate(bin_identities):
                if bin >= 0:
                    output_data[bin] += input_data[index]

            return output_data

        else:
            # Use a mapping based on meshes

            # Either create de-cache the appropriate mesh
            # Why not use a hash? Hashing takes time, equality checks are pretty fast, need to check equality
            # when there is a hit anyway in case of very rare chance of collision, hits are the most common case,
            # we want it to work 100% of the time, not 99.9999%
            if self._input_cache is not None and np.all(self._input_cache.input_coordinates == input_coordinates):

                input_coordinate_mesh = self._input_cache.input_coordinates_mesh
                merge_data = self._input_cache.merged_mesh_data

            else:
                # Calculate mesh data
                input_coordinate_mesh = voronoi_mesh(input_coordinates[:,0], input_coordinates[:, 1])
                self._data_mesh_cache = input_coordinate_mesh

                merge_data = meshmerge(self.bin_mesh, input_coordinate_mesh)

                # Cache mesh data
                self._input_cache = CacheData(
                    input_coordinates=input_coordinates,
                    input_coordinates_mesh=input_coordinate_mesh,
                    merged_mesh_data=merge_data)

            merged_mesh, merged_to_output, merged_to_input = merge_data

            # Calculate values according to the order parameter
            t0 = time.time()
            if order == 0:
                # Based on the overlap of cells only

                input_areas = input_coordinate_mesh.areas
                output = np.zeros(self.bin_mesh.n_cells, dtype=float)

                for input_index, output_index, area in zip(merged_to_input, merged_to_output, merged_mesh.areas):
                    if input_index == -1 or output_index == -1:
                        # merged region does not correspond to anything of interest
                        continue

                    output[output_index] += input_data[input_index] * area / input_areas[input_index]

                print("Main calc:", time.time() - t0)

                return output

            elif order == 1:
                # Linear interpolation requires the following relationship with the data,
                # as the input data is the total over the whole input cell, the linear
                # interpolation requires continuity at the vertices, and a constraint on the
                # integral.
                #
                # We can take each of the input points, and the associated values, and solve a system
                # of linear equations that gives a total value.

                raise NotImplementedError("1st order (linear) interpolation currently not implemented")

            else:
                raise ValueError(f"Expected order to be in {self.allowable_orders}, got {order}")

    def sum(self, x: np.ndarray, y: np.ndarray, data: np.ndarray, order: int = 0) -> np.ndarray:
        """ Return the summed data in the output bins """
        return self._calculate(np.array((x.reshape(-1), y.reshape(-1))).T, data.reshape(-1), order)

    def error_propagate(self, input_coordinates: np.ndarray, data: np.ndarray, errors) -> np.ndarray:
        raise NotImplementedError("Error propagation not implemented yet")

    def resolution_propagate(self, input_coordinates: np.ndarray, data: np.ndarray, errors) -> np.ndarray:
        raise NotImplementedError("Resolution propagation not implemented yet")

    def average(self, x: np.ndarray, y: np.ndarray, data: np.ndarray, order: int = 0) -> np.ndarray:
        """ Return the averaged data in the output bins """
        return self._calculate(np.array((x, y)).T, data.reshape(-1), order) / self.bin_mesh.areas

