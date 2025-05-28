***************************
Preparation of initial data
***************************

Active learning cycle can be initiated from one structure (e.g., transition state) or from existing data sets in ``.npz`` or ``.xyz`` formats. 

---------------------------
Example 1: Loading xyz data
---------------------------

You can train MLIP for existing dataset. The structures can be loaded in xyz format as:

.. code-block:: python

   import mlptrain as mlt

   data = mlt.ConfigurationSet()
   data.load_xyz('data_set.xyz', charge = 0, mult = 1)


This function assumes that all data in your dataset have the same charge and multiplicity. If it is not the case, you can load data with different charges separately.


.. code-block:: python

   import mlptrain as mlt

   data = mlt.ConfigurationSet()
   data.load_xyz('water.xyz', charge = 0, mult = 1)
   data.load_xyz('magnesium_aqua_complex.xyz', charge = -2, mult = 1)

This would allow you to compute *ab initio* labels with correct charges/multiplicities for your data.

If you already have data labeled by reference, this can be loaded as well. However, the data need to be provided in extended xyz format, such as::

            2 
            Lattice="100.000000 0.000000 0.000000 0.000000 100.000000 0.000000 0.000000 0.000000 100.000000"
            energy=-11581.02323085 Properties=species:S:1:pos:R:3:forces:R:3
            C   0.00000   0.00000  0.00000    0.00000    0.00000    0.00000
            O   1.00000   1.00000  1.00000   -1.00000    1.00000    1.00000

where energy and forces are reference data in eV.

You can than load the energies and forces as:

.. code-block:: python

   import mlptrain as mlt

   data = mlt.ConfigurationSet()
   data.load_xyz('data_set.xyz', charge = 0, mult = 1, load_energies = True, load_forces = True)









