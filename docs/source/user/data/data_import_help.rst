.. data_import_help.rst

.. _Loading_data:

Loading data
============

To import data using the scripting interface, ensure the python environment is correctly set up and activated.

    | (sasdata) $ python
    | >>> from sasdata.dataloader.loader import Loader
    | >>> loaded_data_sets = Loader().load(path="/path/to/file.ext") # A list of Data objects
    | >>> loaded_data_set = loaded_data_sets[0] # The first data object

Please read more on the supported :ref:`formats`.

For more information on the data structure, please see the documentation for
`Data1D <../../dev/generated/sasdata.dataloader.html#sasdata.dataloader.data_info.Data1D>`_ for 1-dimensional data and
`Data2D <../../dev/generated/sasdata.dataloader.html#sasdata.dataloader.data_info.Data2D>`_ for 2-dimensional data.

.. ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ

.. note::  This help document was last modified by Jeff Krzywon, 29Sep2023
