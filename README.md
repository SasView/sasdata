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

## Usage

Loading data sets:

    (sasdata) $ python
    >>> from sasdata.dataloader.loader import Loader
    >>> loader_module = Loader()
    >>> loaded_data_sets = loader_module.load(path="/path/to/file.ext")

 - The Loader() class is not callable and must be instantiated prior to use.
 - The load() method returns a list of Data1/2D objects as loaded from the specified path.

Saving loaded data:

    >>> loaded_data_set = loaded_data_sets[0]
    >>> loader.save(path='/path/to/new/file.ext', data=loaded_data_set, format=None)

 - The save() method accepts three (3) arguments:
   - path: The file name and path to save the data.
   - data: A Data1D or Data2D object.
   - format (optional): The expected file extension for the file. Options include:
     - .xml: for the canSAS XMl format
     - .h5: for the NXcanSAS format
     - .txt: for the multi-column ascii format
     - .csv: for a comma delimited text format.
 - The file extension specified in the save path will be superseded by the format value.