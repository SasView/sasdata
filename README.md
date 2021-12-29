# Sasdata

A package for loading and handling reduced small angle scattering data.

The data loaders provided are usable as a standalone package, or in conjunction with the sasview analysis package.

### *This stand-alone package is a work in progress.*

## Install
The easiest way to use sasdata is by using [SasView](http://www.sasview.org).

You can also install sasdata as a standalone package in python. Use
[miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [anaconda](https://www.anaconda.com/)
to create a python environment with the sasdata dependencies::

    $ conda create -n sasdata -c conda-forge numpy lxml h5py

The option ``-n sasdata`` names the environment sasdata, and the option
``-c conda-forge`` selects the conda-forge package channel because pyopencl
is not part of the base anaconda distribution.

Activate the environment and install sasdata::

    $ conda activate sasdata
    (sasdata) $ pip install sasdata

View the latest release on the [sasdata pypi page](https://pypi.org/project/sasdata/).

## Usage

Loading data sets:

    (sasdata) $ python
    >>> from sasdata.dataloader.loader import Loader
    >>> loader_module = Loader()
    >>> loaded_data_sets = loader_module.load("/path/to/file.ext")
