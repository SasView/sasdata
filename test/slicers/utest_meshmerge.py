"""
Tests for mesh merging operations.

It's pretty hard to test componentwise, but we can do some tests of the general behaviour
"""

from sasdata.data_util.slicing.meshes.meshmerge import meshmerge
from test.slicers.meshes_for_testing import (
    grid_mesh, shape_mesh, expected_grid_mappings, expected_shape_mappings)


def test_meshmerge_mappings():

    combined_mesh, grid_mappings, shape_mappings = meshmerge(grid_mesh, shape_mesh)

    for triangle_cell, grid_cell in expected_grid_mappings:
        assert grid_mappings[triangle_cell] == grid_cell

    for triangle_cell, shape_cell in expected_shape_mappings:
        assert shape_mappings[triangle_cell] == shape_cell

