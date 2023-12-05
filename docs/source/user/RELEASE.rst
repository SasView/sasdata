.. RELEASE.rst

.. _Release_Notes:

Release Notes
=============

.. note:: In Windows use [Alt]-[Cursor left] to return to the previous page

.. toctree::
   :maxdepth: 1

Features
========
Wheel, egg, and tar.gz files are available on `pypi <https://pypi.org/project/sasdata/>`_.

New in Version 0.8.1
--------------------
This is a point release to fix a build issue. The `sasdata.data_utils` package was omitted from setup.py causing an
import issue in a separate repository.

New in Version 0.8
------------------
This release brings sasdata into parity with the matching `SasView <https://github.com/SasView/sasview/>`_ data
elements. With this release, the master branch will be locked and all changes will need to be made using pull requests.

New in Version 0.7
------------------
This is the first official release of the sasdata package. This is a stand-alone package with the data
import/export/manipulation available through the `SasView <https://github.com/SasView/sasview/>`_ application.