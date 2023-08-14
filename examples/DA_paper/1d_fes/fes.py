import sys
import mlptrain as mlt
import numpy as np
from mlptrain.box import Box
from mlptrain.log import logger

mlt.Config.n_cores = 10

if __name__ == '__main__':
    us = mlt.UmbrellaSampling(zeta_func=mlt.AverageDistance((1,12), (6,11)),
                              kappa=10)
    temp = 300

    neb = mlt.ConfigurationSet()
    neb.load_xyz(filename='neb_optimised.xyz', charge=0, mult=1)
    
    irc = mlt.ConfigurationSet()
    for config in neb:
        config.box = Box([18.5, 18.5, 18.5])
        irc.append(config)

    r112_reactant = np.linalg.norm(irc[0].atoms[1].coord-irc[0].atoms[12].coord)
    r611_reactant = np.linalg.norm(irc[0].atoms[6].coord-irc[0].atoms[11].coord)

    r112_product = np.linalg.norm(irc[-1].atoms[1].coord-irc[-1].atoms[12].coord)
    r611_product = np.linalg.norm(irc[-1].atoms[6].coord-irc[-1].atoms[11].coord)

    logger.info(f'average bond length in reactant is {(r112_reactant+r611_reactant)/2}')
    logger.info(f'average bond length in product is {(r112_product+r611_product)/2}')

    irc.reverse()  # Go product -> reactant, the NEB path is from reactant -> product

    water_mol = mlt.Molecule(name='h2o.xyz')
    TS_mol = mlt.Molecule(name='cis_endo_TS_wB97M.xyz')

    system = mlt.System(TS_mol, box=Box([100, 100, 100]))
    system.add_molecules(water_mol, num=200)

    endo = mlt.potentials.ACE('endo_in_water_ace_wB97M', system)

    us.run_umbrella_sampling(irc,
                             mlp=endo,
                             temp=temp,
                             interval=5,
                             dt=0.5,
                             n_windows=15,
                             init_ref=1.55,
                             final_ref=4,
                             ps=10)
    us.save('wide_US')

    # Run a second, narrower US with a higher force constant
    us.kappa = 20
    us.run_umbrella_sampling(irc,
                             mlp=endo,
                             temp=temp,
                             interval=5,
                             dt=0.5,
                             n_windows=15,
                             init_ref=1.7,
                             final_ref=2.5,
                             ps=10)

    us.save('narrow_US')

    total_us = mlt.UmbrellaSampling.from_folders('wide_US', 'narrow_US',
                                                 temp=temp)
    total_us.wham()
