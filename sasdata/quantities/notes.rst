Mutability
----------

DataSets: Immutable
Quantities: Immutable
Units: Hard coded

Quantity methods
----------------

in_* methods return numbers/arrays in a given unit system
to_* converts to different units


Identifying of Quantities
--------------------

There are two choices when it comes to keeping track of quantities for error propagation.
Either we give them names, in which case we risk collisions, or we use hashes, which can potentially
have issues with things not being identified correctly.

The decision here is to use hashes of the data, not names, because it would be too easy to
give different things the same name.