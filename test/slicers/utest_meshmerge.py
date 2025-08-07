"""
Tests for mesh merging operations.

It's pretty hard to test componentwise, but we can do some tests of the general behaviour
"""

from sasdata.slicing.meshes.meshmerge import meshmerge
from test.slicers.meshes_for_testing import expected_grid_mappings, expected_shape_mappings, grid_mesh, shape_mesh


def test_meshmerge_mappings():
    """ Test the output of meshmerge is correct

    IMPORTANT IF TESTS FAIL!!!... The docs for scipy.spatial.Voronoi and Delaunay
    say that the ordering of faces might depend on machine precession. Thus, these
    tests might not be reliable... we'll see how they play out
    """

    import sys
    if sys.platform == "darwin":
        # It does indeed rely on machine precision, only run on windows and linux
        return

    combined_mesh, grid_mappings, shape_mappings = meshmerge(grid_mesh, shape_mesh)

    for triangle_cell, grid_cell in expected_grid_mappings:
        assert grid_mappings[triangle_cell] == grid_cell

    for triangle_cell, shape_cell in expected_shape_mappings:
        assert shape_mappings[triangle_cell] == shape_cell

