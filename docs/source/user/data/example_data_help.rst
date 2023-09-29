.. example_data_help.rst

.. _example_data_help:

Example Data
============

Example data sets are included as a convenience to our users. The data sets are organized based on their data structure;
1D data (ie, I(Q)), 2D data (ie, I(Qx,Qy)), image data (eg, TIFF files), and SESANS data, to name a few.

1D data sets EITHER a) have at least two columns of data with I(abs. units) on the y-axis and Q on the x-axis, OR b)
have I and Q in separate files. Data in the latter format (/convertible_files) need to be converted to a single file format
with the File Converter tool before SasView will analyse them.

2D data sets are data sets that give the reduced intensity for each QX, Qx point. Depending on the file extension,
uncertainty and metadata may also be available.

Image data sets are designed to be read by the Image Viewer tool.

SESANS data sets primarily contain the neutron polarisation as a function of the spin-echo length.
