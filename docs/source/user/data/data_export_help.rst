.. data_export_help.rst

.. _Exporting_data:

Exporting data
==============

The `Loader().save()` method accepts 3 arguments; `path`, which is the file path to save the file into, `data`, which is the
`Data1D` or `Data2D` object, and, optionally, `ext`, a file extension. If an extension is passed to `save`, any file extension
in the file path will be superseded. If no file extension is given in the filename or format, a ValueError will be thrown.

Save `format` options are limited to:
  * `.xml`: for the canSAS XML format
  * `.h5`: for the NXcanSAS format
  * `.txt`: for the multi-column ascii format
  * `.csv`: for a comma delimited text format

.. list-table:: Save argument examples and associated data output
   :header-rows: 1

   * - filename
     - format
     - saved file name
     - saved file format
   * - 'mydata'
     - '.csv'
     - mydata.csv
     - CSV format
   * - 'mydata.xml'
     - None
     - mydata.xml
     - canSAS XML format
   * - 'mydata.xml'
     - '.csv'
     - mydata.xml.csv
     - CSV format
   * - 'mydata'
     - None
     - N/A
     - raises `ValueError`

To export data using the scripting interface, ensure the python environment is correctly set up and activated.

.. code-block:: RST

    (sasdata) $ python
    >>> from sasdata.dataloader.loader import Loader
    >>> loader_module = Loader()
    >>> loaded_data_sets = loader_module.load(path="/path/to/imported/file.ext")
    >>> loaded_data_set = loaded_data_sets[0]
    >>> loader_module.save(path='/path/to/file/exported/filename.ext', data=loaded_data_set)

Please read more on the supported :ref:`formats`.

For more information on the native data structure, please see the documentation for
`Data1D <../../dev/generated/sasdata.dataloader.html#sasdata.dataloader.data_info.Data1D>`_ for 1-dimensional data and
`Data2D <../../dev/generated/sasdata.dataloader.html#sasdata.dataloader.data_info.Data2D>`_ for 2-dimensional data.

.. ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ

.. note::  This help document was last modified by Jeff Krzywon, 29Sep2023
