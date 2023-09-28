.. data_explorer_help.rst

.. This is a port of the original SasView html help file to ReSTructured text
.. by S King, ISIS, during SasView CodeCamp-III in Feb 2015.

.. _Loading_data:

Loading data
============

To load data using the scripting interface,

    (sasdata) $ python
    >>> from sasdata.dataloader.loader import Loader
    >>> loaded_data_sets = Loader().load(path="/path/to/file.ext")
    >>> loaded_data_set = loaded_data_sets[0]
    >>> loader_module.save(path='/path/to/new/file.ext', data=loaded_data_set, format=None)

.. ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ

.. note::  This help document was last changed by Jeff Krzywon, 28Sep2023
