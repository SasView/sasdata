"""Tests for the collection of selectable dataset types (issue #265)."""

from sasdata.dataset_types import DatasetType, angle_dim, dataset_types, one_dim, sesans, three_dim, two_dim


def test_dataset_types_order():
    # The order is fixed, so that the dataset types are always shown in the same order (it used to be a set).
    assert list(dataset_types) == [one_dim.name, two_dim.name, angle_dim.name, sesans.name]


def test_dataset_types_lookup_by_name():
    for dataset in [one_dim, two_dim, angle_dim, sesans]:
        assert dataset_types[dataset.name] is dataset
    assert all(isinstance(dataset, DatasetType) for dataset in dataset_types.values())


def test_three_dim_not_included():
    assert three_dim.name not in dataset_types


def test_names_can_still_be_used_as_before():
    # Iterating, and checking if a name is included, work as they did when this was a set of names.
    assert set(dataset_types) == {one_dim.name, two_dim.name, angle_dim.name, sesans.name}
    assert "1D I vs Q" in dataset_types
