# Sasdata

A package for importing and exporting reduced small angle scattering data.

The data loaders provided are usable as a standalone package, or in conjunction with the sasview analysis package.

## Install
The easiest way to use sasdata is by using [SasView](http://www.sasview.org).

View the latest release on the [sasdata pypi page](https://pypi.org/project/sasdata/) and install using `pip install sasdata`.

To run sasdata from the source, create a python environment using python 3.8 or higher and install all dependencies
 - Using a [python virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)::

       $ python -m venv sasdata
       $ .\sasdata\Scripts\activate (Windows) -or- source sasdata/bin/activate (Mac/Linux)
       (sasdata) $ python -m pip install -r requirements.txt
       (sasdata) $ python /path/to/sasdata/setup.py clean build
       (sasdata) $ python -m pip install .
 
 - Using [miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [anaconda](https://www.anaconda.com/)::
   
       $ conda create -n sasdata
       $ conda activate sasdata
       (sasdata) $ python -m pip install -r requirements.txt
       (sasdata) $ python /path/to/sasdata/setup.py clean build
       (sasdata) $ python -m pip install .

## Data Formats

The `Loader()` class is directly callable so a transient call can be made to the class or, for cases where repeated calls
are necessary, the `Loader()` instance can be assigned to a python variable.

The `Loader.load()` method accepts a string or list of strings to load a single or multiple data sets simultaneously. The
strings passed to `load()` can be any combination of file path representations, or URIs. A list of `Data1D/Data2D`
objects is returned. An optional `format` parameter can be passed to specify the expected file extension associated with 
a reader. If format is passed, it must either be a single value, or a list of values of the same length as the file path list.

- Load `format` options include:
  - `.xml`: [canSAS XML](https://www.cansas.org/formats/canSAS1d/1.1/doc/index.html) format
  - `.h5`, `.hdf`, `.hdf5`, `.nxs`: [NXcanSAS](https://manual.nexusformat.org/classes/applications/NXcanSAS.html) format
  - `.txt`: Multi-column ascii format
  - `.csv`: Comma delimited text format
  - `.ses`, `.sesans`: [Multi-column SESANS](https://www.sasview.org/docs/user/qtgui/MainWindow/data_formats_help.html#d-sesans-format) data
  - `.dat`: [2D NIST](https://github.com/sansigormacros/ncnrsansigormacros/wiki/NCNROutput2D_QxQy) format
  - `.abs`, `.cor`: 1D NIST format for SAS and USAS
  - `.pdh`: Anton Paar reduced SAXS format

The `save()` method accepts 3 arguments; the file path to save the file as, a `Data1D` or `Data2D` object, and, optionally, 
a file extension. If an extension is passed to `save`, any file extension in the file path will be superseded. If no file
extension is given in the filename or format, a ValueError will be thrown.

- Save `format` options include:
  - `.xml`: for the canSAS XML format
  - `.h5`: for the NXcanSAS format
  - `.txt`: for the multi-column ascii format
  - `.csv`: for a comma delimited text format

Save argument examples and data output:

| filename     | format | saved file name | saved file format |
|--------------|--------|-----------------|-------------------|
| 'mydata'     | '.csv' | mydata.csv      | CSV format        |
| 'mydata.xml' | None   | mydata.xml      | canSAS XML format |
| 'mydata.xml' | '.csv' | mydata.xml.csv  | CSV format        |
| 'mydata'     | None   | -               | raise ValueError  |

More information on the recognized data formats is available on the 
[sasview website](https://www.sasview.org/docs/user/qtgui/MainWindow/data_formats_help.html).

## Usage

### Loading and saving data sets using a fixed Loader instance:

    (sasdata) $ python
    >>> from sasdata.dataloader.loader import Loader
    >>> loader_module = Loader()
    >>> loaded_data_sets = loader_module.load(path="/path/to/file.ext")
    >>> loaded_data_set = loaded_data_sets[0]
    >>> loader_module.save(path='/path/to/new/file.ext', data=loaded_data_set, format=None)

### Loading and saving data sets using a transient Loader instance (more scriptable):

    (sasdata) $ python
    >>> from sasdata.dataloader.loader import Loader
    >>> loaded_data_sets = Loader().load(path="/path/to/file.ext")
    >>> Loader().save(path='/path/to/new/file.ext', data=loaded_data_sets[0], format=None)
