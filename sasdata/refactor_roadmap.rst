* percentages updated last on 08/10/2025

SasData Side
============

* Load data into memory as SasData objects

  * From different sources

    * Text
    * ASCII (95%)
    * XML   (95%)
    * HDF5  (95%)

  * known mistakes (living object)

* Have calculations of errors and tracking of objects

  * Low level implementation of error prop (95%)
  * Linking of data (10%)
  * Independent source identification (50%, but see above)

    * Make a decision (0%)
    * Carefully document said decision (0%)

  * Units (95%)

    * Unit conversion (100%)
    * Unit parsing (95%)
    * Big file of useful units and not so useful units (80%)
    * Printing units (95%)

  * Operations on Quantities

    * Arithmetic operations (95%)
    * Special and not so special functions (50%)
    * Linear algebra (30%)

* Operation on datasets

  * Rebinning (35%)
  * Slicing backend [integration] (see above)
  * Adding extra annotations (0%)

* Trends

  * Construct trend (70%)
  * Interpolate axes (80%)

* Save data

  * Into something readable (JSON) (95%)
  * Into HDF5 (95%)

Develop a Rigorous Testing Framework For Critical Objects
=========================================================

Currently lots of tests, but we should be more systematic.




SasView Side (Integration)
==========================

* ASCII loader interface (90%)
* Data explorer refactor (30%)

  * Represent data in GUI (85%)
  * Represent plots in GUI (0%)
  * Represent links between data in GUI (0%)
  * Represent perspectives in GUI (0%)
  * Represent trends in GUI (0%)

* Trends

  * Suggest metadata (65%)
  * Present options to user (0%)

* Perspectives

  * Make them accept new data object (25%)
  * Batch stuff should become trend stuff (0%)

* Slicing

  * Refactor slicers for new backend (0%)


`_________|o=o\______` -> this way

