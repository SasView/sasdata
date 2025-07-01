# Sasdata

A package for importing and exporting reduced small angle scattering data.

The data loaders provided are usable as a standalone package, or in conjunction with the sasview analysis package.

## Install
The easiest way to use sasdata is by using [SasView](http://www.sasview.org).

View the latest release on the [sasdata pypi page](https://pypi.org/project/sasdata/) and install using `pip install sasdata`.

To run sasdata from the source, create a python environment using python 3.10 or higher and install all dependencies
 - Using a [python virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)::

       $ python -m venv sasdata
       $ .\sasdata\Scripts\activate (Windows) -or- source sasdata/bin/activate (Mac/Linux)
       (sasdata) $ python -m pip install -e .
 
 - Using any [anaconda](https://www.anaconda.com/) distribution::
   
       $ conda create -n sasdata
       $ conda activate sasdata
       (sasdata) $ python -m pip install -e .

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

## Example Data

A number of data files are included with this package available in `sasdata.example_data`.

- Each subdirectory has a specific type of data.
  - `1d_data`: 1-dimensional SAS data. A few examples are `apoferritin.txt` which is SANS from apoferritin, 3 files starting with `AOT_` that are contrast variations for a mircroemulsion, and `latex_smeared.xml` with SANS and USANS data for spherical latex particles.
  - `2d_data`: 2-dimensional SAS data. Examples include 3 `P123_....dat` files for a polymer concentration series.
  - `convertibles_files`: A series of data sets that can be converted via the data conversion tool in the `sasdata.file_converter` package.
  - `dls_data`: *NOTE* Not loadable by sasdata. Two example DLS data sets that will be loadable in a future release.
  - `image_data`: Image file loadable from `sasdata.dataloader.readers.tiff_reader`. The files are all the same image, but in different image formats.
  - `sesans_data`: SESANS data sets. `sphere_isis.ses` is spin-echo SANS from a sample with spherical particles.
- To directly access this data via a python prompt, `import data_path from sasdata` returns the absolute path to `sasdata.example_data`

## Usage

### Accessing example data

    (sasdata) $ python
    >>> from sasdata import data_path
    >>> data = Loader().load(os.path.join(data_path, '1d_data', 'apoferritin.txt'))

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
