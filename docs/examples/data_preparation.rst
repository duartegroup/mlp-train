***************************
Preparation of initial data
***************************

Active learning cycle can be initiated from one structure (e.g., transition state) or from existing data sets in ``.npz`` or ``.xyz`` formats. 

----------------
Loading xyz data
----------------

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
   data.load_xyz('magnesium_aqua_complex.xyz', charge = 2, mult = 1)

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

   
------------------------------------------
Label data with reference energy and force
------------------------------------------

To train MLIP, the structures need to be labelled by suitable reference. This is typically energy and forces from electronic structure computations.
``mlp-train`` currently supports computing the reference data by ``Orca``, ``Gaussian`` or ``xtb``. Below, we discuss the examples how to use them: 

ORCA
----

To be able to use ``Orca``, you first need to get the suitable binary from: https://www.faccts.de/orca/ and add it to your $PATH. Orca is free for personal and academic use, but you will need to register.

When you have the binary file, you can use `Orca` to label the data. Assume that you load the ``data_set.xyz`` file as discussed in previous example. 

First, select the level of theory, i.e., the DFT functional and basis set and specify `Engrad` keyword to compute both energy and gradients.

.. code-block:: python

   mlt.Config.orca_keywords = ['PBE', 'def2-SVP', 'EnGrad']


You can provide more keywords, using the same synthax as you would use with `Orca`. For example:

.. code-block:: python

    mlt.Config.orca_keywords = [
      'PBE0',
      'D3BJ',
      'def2-TZVP',
      'def2/J',
      'RIJCOSX',
      'EnGrad',
      'CPCM(Water)'
    ]

More advanced settings can be provided in scf_block, following the same structure as normal Orca input:

.. code-block:: python

    scf_block= (
     '\n%scf\n'
     'MaxIter 1000\n'
     'DIISMaxEq 15\n'
     'end\n'
     '%cpcm\n'
     'smd true\n'
     'SMDSolvent "Acetonitrile"\n'
     'end\n'
    )

    mlt.Config.orca_keywords = ['UKS','r2SCAN-3c','EnGrad','VeryTightSCF', 'defgrid3', 'NoTrah', 'SlowConv', scf_block]


Afterwards, you can run single point computations over the loaded ConfigurationSet:

.. code-block:: python

   data.single_point(method='orca')
   data.save_npz('data_set_labelled.npz')

Gaussian
--------

Unlike Orca, Gaussian requires paid licenece. More information can be found: https://gaussian.com/

The synthax is very similar as in previous case:

.. code-block:: python

   mlt.Config.gaussian_keywords = ['PBEPBE', 'Def2SVP', 'Force(NoStep)', 'integral=ultrafinegrid']

You can choose from Gaussian 09 (`'g09'`) or Gaussian 16 (`'g16'`). For Gaussian 16, the synthax would be as follows:

.. code-block:: python

   data.single_point(method='g16')
   data.save_npz('data_set_labelled.npz')

xTB
---

Finally, you can label your data using GFN2-xTB semiempirical method. However, the training MLIP on this level is recomanded only for testing of the workflow.

.. code-block:: python

   data.single_point(method='g16')
   data.save_npz('data_set_labelled.npz')




