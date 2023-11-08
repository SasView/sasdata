.. data_import_help.rst

.. _Importing_data:

Importing data
==============

The `Loader()` class is directly callable so a transient call can be made to the class or, for cases where repeated calls
are necessary, the `Loader()` instance can be assigned to a python variable.

The `Loader.load()` method accepts a string or list of strings to load a single or multiple data sets simultaneously. The
paths passed to `load()` can be any combination of `Path` objects, file path like string representations, or URIs. A list of `Data1D/Data2D`
objects is returned. An optional `format` parameter can be passed to specify the expected file extension associated with
a reader. If format is passed, it must either be a single value, or a list of values of the same length as the file path list.

To import data using the scripting interface, ensure the python environment is correctly set up and activated.

.. code-block:: RST

    (sasdata) $ python
    >>> from sasdata.dataloader.loader import Loader
    >>> loaded_data_sets = Loader().load(path="/path/to/file.ext") # A list of Data objects
    >>> loaded_data_set = loaded_data_sets[0] # The first data object

Please read more on the supported :ref:`formats`.

For more information on the data structure, please see the documentation for
`Data1D <../../dev/generated/sasdata.dataloader.html#sasdata.dataloader.data_info.Data1D>`_ for 1-dimensional data and
`Data2D <../../dev/generated/sasdata.dataloader.html#sasdata.dataloader.data_info.Data2D>`_ for 2-dimensional data.

.. ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ

.. note::  This help document was last modified by Jeff Krzywon, 29Sep2023
