************************************************
Generating dataset for reaction in the gas phase
************************************************

In this tutorial, we will look into using active learing (AL) to generate data set for a simple organic chemical reaction in the gas phase. We will model the Diels-Alder (DA) reaction between cyclopentadiene (CP) and methyl vinyl ketone (MVK) using MACE potential.

.. figure:: ../images/downhill/DA_scheme.pdf
   :alt: Scheme of the Diels-Alder reaction between cyclopentadiene and methyl vinyl ketone.
   :width: 80%
   :align: center

   Fig. 1: Scheme of the Diels-Alder reaction between cyclopentadiene and methyl vinyl ketone.

To accurately model chemical reactivity, MLIP needs to be trained on data set containig reactants, products and the reaction pathway between the point crossing the transition state (TS) region. One of the strategies how to ensure that whole reaction path is covered in the training set is to start the AL from the TS. The subsequent dynamics will ensure that the structure will follow the gradient downhill on the potential energy surface (PES) forward towards products or back to reactants. 

We will start from the TS optimised at PBE0/def2-svp level of theory, which is saved in ``cis_endo_TS_PBE0.xyz`` file.

.. code-block:: python
 
  22

  C   -2.45625042994264      4.22697136718373     -0.69050566209957
  C   -3.28702195379458      3.16527993919028     -0.02209098677282
  C   -2.41028766314595      2.14322126716790      0.34022614745311
  H   -4.18631817798479      3.39651295844350      0.53629598217147
  C   -1.20817318676536      2.29480857602016     -0.36982113539345
  H   -2.66413749952356      1.29424573410959      0.95978388172974
  C   -1.28146913102924      3.42909805269927     -1.13984450034102
  H   -0.40867019996620      1.56728893158943     -0.38835776572309
  H   -0.50683361420779      3.80598439935262     -1.79369703691900
  H   -2.11400085795616      4.92541367601641      0.08909189132240
  H   -2.95734970838890      4.81263532467316     -1.45923411982257
  C   -3.10205484705813      2.37368598258547     -2.68346342635020
  C   -4.11637859827911      2.44256568361059     -1.74418593887422
  H   -4.85151124593427      3.23644849500096     -1.82006001153415
  H   -4.47736922293900      1.50457689648386     -1.33925710572378
  C   -2.36389642614390      1.13176739960370     -2.83295879682348
  H   -2.91472881437345      3.18240648333903     -3.37940748993174
  O   -2.50989176905760      0.20606095908980     -2.04760734145413
  C   -1.42166214759852      1.00292166731278     -4.00365141692310
  H   -1.03881265683699      1.96596828196554     -4.34659501197343
  H   -1.96765811314529      0.54168812578948     -4.83288296604129
  H   -0.59713790022863      0.34139508057279     -3.73891706567561


The simple code to run the AL from TS can be written as follows:

.. code-block:: python

  import mlptrain as mlt
  
  mlt.Config.n_cores = 10
  mlt.Config.orca_keywords = ['PBE0', 'def2-SVP', 'EnGrad']
  mlt.Config.mace_params['calc_device'] = 'cuda'
  
  if __name__ == '__main__':
      system = mlt.System(
          mlt.Molecule('cis_endo_TS_PBE0.xyz', charge=0, mult=1), box=None
      )
  
      mace = mlt.potentials.MACE('endo_ace', system=system)
  
      mace.al_train(
          method_name='orca',
          temp=500,
          max_active_time=1000,
          fix_init_config=True,
          keep_al_trajs=True,
      )
  
      # Run some dynamics with the potential
      trajectory = mlt.md.run_mlp_md(
          configuration=system.configuration,
          mlp=mace,
          fs=200,
          temp=300,
          dt=0.5,
          interval=10,
      )
  
      # and compare, plotting a parity diagram and E_true, $\Delta$ E and $\Delta$ F
      trajectory.compare(mace, 'orca')

First part of the code provides definition of the level of theory needed, number of CPUs and CUDA acceleration. After, we load the structure of the TS and set its charge and multiplicity, which will be used later in the QM calculations.

For this system, we select MACE for the training.

Finally, we can define the active learning loop:

.. code-block:: python
  
   mace.al_train(
        method_name='orca',
        temp=500,
        max_active_time=1000,
        fix_init_config=True,
        keep_al_trajs=True,
    )


First, we set the electronic structure code used in AL (``method_name``) to `'orca'`. We then fix the initial configuration in the AL ``selectfont fix_init_config=True`` to ensure that each AL cycle will start from the TS structure, i.e., the downhill sampling will be used in every run. Finally, we will set ``keep_al_trajs=True``, to save the trajectories sampled during each AL for future reference.

After AL, we can check the coverage of the reaction space by the training set by plotting the data based on the collective variable, defined as $\frac{r_1 + r_2}{2}$, where $r_1$ and $r_2$ are the two bonds formed during the Diels-Alder reaction.

.. code-block:: python
  
  import numpy as np
  from matplotlib import rc
  import matplotlib.pyplot as plt
  import ase.io as aio

  rc("text", usetex=True)

  data = aio.read("endo_ace_al.xyz", index=":")


  collective_variable = []

  for structure in data:
    r1 = structure.get_distance(1, 12)
    r2 = structure.get_distance(6, 11)
    collective_variable.append(0.5 * (r1 + r2))


  x = np.arange(0, len(collective_variable))

  clas = np.where(
    np.array(collective_variable) < 1.8,
    "PS",
    np.where(np.array(collective_variable) > 2.8, "RS", "TS"),
  )

  cdict = {"RS": "red", "TS": "blue", "PS": "black"}

  fig, ax = plt.subplots()
  for c in np.unique(clas):
    ix = np.where(clas == c)
    ax.scatter(x[ix], np.array(collective_variable)[ix], c=cdict[c], label=c, s=10)

  plt.xlabel("Index")
  plt.ylabel(r"$\frac{(r_1+r_2)}{2}$")
  plt.title("Data points")
  plt.legend()

  plt.savefig("r12_dataset.pdf", bbox_inches="tight")


.. figure:: ../images/downhill/r12_dataset.pdf
   :alt: Classification of the structures from the dataset.
   :width: 80%
   :align: center

   Fig. 1: Classification of the structures from the dataset based on r1 + r2 .  

You can see that the first 10 data points are very similar to each other - these correspond to initial data set of 10 distorted TS structures. Afterwards, the downhill sampling in AL generates structures of both reactants and products.

We can now check the performance of the MACE over a short 200-fs validation trajectory. 

The MAD in energy is 29 meV, corresponding to 1.32 meV/atom. MAD in forces is 77 meV/\AA. These errors are realitively high, suggesting that we might need to set the time in the AL longer than in the current settings, which is only 1 ps. 

The resulting potential can now be used for other simulations. For instance, we can run an umbrella sampling (US) simulation to compute the free energy barrier of the reaction.

``mlp-train`` includes its own US implementation, which automatically defines the windows.

The example script can look like this:

.. code-block:: python
  
  import mlptrain as mlt
  import numpy as np
  from mlptrain.box import Box
  from mlptrain.log import logger

  mlt.Config.mace_params["calc_device"] = "cuda"

  if __name__ == "__main__":
    us = mlt.UmbrellaSampling(zeta_func=mlt.AverageDistance((1, 12), (6, 11)), kappa=10)

    irc = mlt.ConfigurationSet()
    irc.load_xyz(filename="irc_IRC_Full_trj.xyz", charge=0, mult=1)

    for config in irc:
        config.box = Box([100, 100, 100])

    irc.reverse()

    TS_mol = mlt.Molecule(name="cis_endo_TS_PBE0.xyz", charge=0, mult=1, box=None)

    system = mlt.System(TS_mol, box=Box([100, 100, 100]))

    endo = mlt.potentials.MACE("endo_ace_stagetwo", system)

    us.run_umbrella_sampling(
        irc,
        mlp=endo,
        temp=300,
        interval=5,
        dt=0.5,
        n_windows=15,
        init_ref=1.55,
        final_ref=4,
        ps=10,
    )
    us.save("wide_US")

    # Run a second, narrower US with a higher force constant
    us.kappa = 20
    us.run_umbrella_sampling(
        irc,
        mlp=endo,
        temp=300,
        interval=5,
        dt=0.5,
        n_windows=15,
        init_ref=1.7,
        final_ref=2.5,
        ps=10,
    )

    us.save("narrow_US")

    total_us = mlt.UmbrellaSampling.from_folders("wide_US", "narrow_US", temp=temp)
    total_us.wham()

