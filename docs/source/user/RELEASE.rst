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

New in Version 0.11.0
---------------------

This release introduces some automated linting with Ruff for developers by @DrPaulSharp, and also fixes a bug in a formula. There has also been some work towards automated PyPi releases from @krzywon.

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
