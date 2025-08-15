SasData Refactor Principles
===========================


Fundamental choices
-------------------

1: Data is Immutable
====================

We want data to be cross referenced and keep track of important information about correlations. Furthermore, to follow
FAIR principles we want a record of this.

It is very hard to do these things if we allow data to be mutable. This doesn't mean you can't copy the contents
of a data and change the copy, then make a new data object.

2: Data is tracked
==================

Again, within the SasData objects, operations are tracked. This allows use to propagate uncertainties correctly
and keep a FAIR record.

3: Data objects are mostly agnostic to their contents
=====================================================

To represent more general kinds of data in uniform way, we have data objects that don't specifically have q/I axes.
All axes are however named, and dimensionality is implicit.

3b: Data types are "duck-typed"
===============================

The way whether we tell data is, for example, Q/I data is by checking the axes names, not by the class.

Note: Some checks of this kind should probably be implemented as utility functions

3c: Errors are bound to quantities
==================================

Errors are bound to the quantities they correspond to, so for example, I and dI are held in a single object.

Note: It is probably incorrect to associate Q and dQ like this, as they don't follow standard error propagation rules,
instead, use supplementary information.


4: Relationship to models is specified in the data class
========================================================

The processing steps needed to convert model outputs to something comparable to the data is included in the `modelling
requirements` section. Making use of this is optional.



5: Trends
=========

Have collections of SasData objects that can be treated together, e.g. as functions of some variable, field, concentration

