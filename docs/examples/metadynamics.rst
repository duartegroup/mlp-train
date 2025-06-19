********************************
Run Metadynamics with MLP model
********************************

Once an MLP model is generated, it can be used to run metadynamics simulution, which is supported in mlp-train interfaced to PLUMED (https://www.plumed.org/doc-v2.9/user-doc/html/index).


---------------------------------------------
Part 1: Run a simple metadynamics simulation
---------------------------------------------

This example assumes that you already have an MLP model (e.g. water.model) and an intial configuration for metadynamics (e.g. water.xyz) in your working directory:

Firstly, the model is loaded by specifying the system and the model:

.. code-block:: python

   import mlptrain as mlt

   system = mlt.System(
       mlt.Molecule("water.xyz", charge=0, mult=1), box=mlt.Box([100,100,100])
   )

   # a random configuration is generated for the latter metadynamics simulation use 
   config = system.random_configuration()

   mace = mlt.potentials.MACE('water', system=system)


Second, you need to specify the collective variables and associated parameter (type of collective variables, atom indexes, and restrain potentials, etc. More information could be found in Documentations>References>Sampling). Here is an example using dihedral angle as CVs:

.. code-block:: python

   import numpy as np

   dih = mlt.PlumedAverageCV(name='dih', atom_groups=[(1,2,3,4)])
   #Note: atomic index here is 0-based

   dih.attach_lower_wall(location=-np.pi, kappa=10000)
   dih.attach_upper_wall(location=np.pi, kappa=10000)

   metad = mlt.Metadynamics(cvs=dih)

Finally, you can run metadynamics simulation using ``run_metadynamics`` function and specify the associated parameters, such as width, height (**in eV**) and biasfactor.

.. code-block:: python

   metad.run_metadynamics(
       configuration=config,
       mlp=mace,
       temp=300,
       width=1.10,
       height=0.01,
       biasfactor=10
   )


The free energy profile can also be plotted using

.. code-block:: python

    metad.plot_fes()
    metad.plot_fes_convergence(stride=10, n_surfaces=5)

---------------------------------------------------------------------
Part 2: Estimate the width and biasfactor
---------------------------------------------------------------------

You can also estimate the suitable width or biasfactors with the functions ``estimate_width`` or ``try_multiple_biasfactors``, respectively. For width estimation, it works by running some small NVT simulatios, selecting the minimum standard deviation as the optimal width across all simulations for each collective variable. For biasfactor, multiple well-tempered metadynamics runs in parallel with the provided biasfactors and the resulting trajectories are plotted for estimating the optimal biasfactor value.

.. code-block:: python

    width = metad.estimate_width(configurations=config, mlp=mace, plot=True)
    metad.try_multiple_biasfactors(
        configuration=config,
        mlp=mace,
        width=width,
        biasfactor=(5,10,15),
        plotted_cvs=dih,
    )




