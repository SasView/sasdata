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

New in Version 0.12.0
----------------------
This release, which coincides with SasView 6.2.0, includes a number of bug fixes and feature enhancements.
It includes a refactor and clean up of `manipulations.py` and `averaging.py` by @ehewins and @jellybean2004,
optimised string concatenation by @mohasarc and a full rotation to units by @rprospero. Thew first part of
CodeScene integration was added by @krzywon. Bug fixes included slicer fixes by @dehoni and a fix for PyPI
upload by @krzywon. 

This release also includes three pull requests from the SasData refactor for developer use.

What's Changed
^^^^^^^^^^^^^^

Feature Enhancements
____________________
* Finished the `manipulations.py` rewrite by @ehewins in https://github.com/SasView/sasdata/pull/47
* Upgraded Python version to 3.12 and fixed linting errors by @DrPaulSharp in https://github.com/SasView/sasdata/pull/166
* Added quantities work from SasData refactor by @DrPaulSharp in https://github.com/SasView/sasdata/pull/167
* Added load/save work from SasData refactor by @DrPaulSharp in https://github.com/SasView/sasdata/pull/168
* Added trend work from SasData refactor by @DrPaulSharp in https://github.com/SasView/sasdata/pull/169
* Added CodeScene integration (Part 1) by @krzywon in https://github.com/SasView/sasdata/pull/182
* Added full rotation to units by @rprospero in https://github.com/SasView/sasdata/pull/187
* Optimised string concatenation using `.join()` by @mohasarc in https://github.com/SasView/sasdata/pull/185
* Updated citation and contributors information by @krzywon in https://github.com/SasView/sasdata/pull/200
* Cleaned up averaging functionality by @jellybean2004 in https://github.com/SasView/sasdata/pull/203


Bug Fixes
_________
* Fixed PyPI upload by @krzywon in https://github.com/SasView/sasdata/pull/160
* Fixed slicers not working for 2D residual plots by @dehoni in https://github.com/SasView/sasdata/pull/171
* Fixed sector averaging for circular regions by @dehoni in https://github.com/SasView/sasdata/pull/174

New in Version 0.10.0
---------------------

This release, which coincides with SasView 6.1.0, includes changes by @llimeht
to the pyproject.toml file to use hatchling as the build backend. There have
also been some minor changes which provide compatibility for the new project
structure of SasView.

Test data has also been added by @butlerpd for use with the Size Distribution perspective.

New in Version 0.9.0
--------------------
This is an enhancement release with updates to the unit conversion routines, the ability to load data from URIs, the
addition of a wedge slicer and other slicer enhancements.

What's Changed
^^^^^^^^^^^^^^

Feature Enhancements
____________________
* Refactor nxsunit by @krzywon in https://github.com/SasView/sasdata/pull/13
* Enable the sector slicing to allow both sides independantly by @butlerpd in https://github.com/SasView/sasdata/pull/36
* Load data from URIs by @krzywon in https://github.com/SasView/sasdata/pull/37
* SasData counterpart to SasView SlicerExtension_1344 by @jack-rooks in https://github.com/SasView/sasdata/pull/61

Bug Fixes
_________
* Fixing Issue #40 (but properly this time) by @ehewins in https://github.com/SasView/sasdata/pull/42
* changed xaxis label for updated SESANS nomenclature from z to delta by @caitwolf in https://github.com/SasView/sasdata/pull/60
* Fix delta in sesans docs by @caitwolf in https://github.com/SasView/sasdata/pull/65

Documentation Changes
_____________________
* Rework readme by @krzywon in https://github.com/SasView/sasdata/pull/15
* Building sasdata documentation by @krzywon in https://github.com/SasView/sasdata/pull/53
* Generate Developer Docs by @krzywon in https://github.com/SasView/sasdata/pull/56

Infrastructure Changes
______________________
* Remove entry_point from setup.py by @krzywon in https://github.com/SasView/sasdata/pull/2
* Dependency cleanup by @krzywon in https://github.com/SasView/sasdata/pull/33
* Move example data to sasdata by @krzywon in https://github.com/SasView/sasdata/pull/49
* CI updates by @krzywon in https://github.com/SasView/sasdata/pull/50
* Restrict lxml to versions less than 5.0 by @krzywon in https://github.com/SasView/sasdata/pull/63
* Update example data by @smk78 in https://github.com/SasView/sasdata/pull/58
* Fix broken unit test(s) by @krzywon in https://github.com/SasView/sasdata/pull/68

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
