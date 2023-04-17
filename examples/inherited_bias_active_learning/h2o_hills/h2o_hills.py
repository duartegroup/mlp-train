import mlptrain as mlt

mlt.Config.n_cores = 10

if __name__ == '__main__':

    # Initialise the system to train

    h2o_system = mlt.System(mlt.Molecule('h2o.xyz'), box=None)

    # Define collective variables and attach upper walls
    # to restrain the system from exploding

    cv1 = mlt.PlumedAverageCV(name='cv1', atom_groups=((0, 1), (0, 2)))
    cv1.attach_upper_wall(location=5, kappa=10000)

    cv2 = mlt.PlumedAverageCV(name='cv2', atom_groups=(1, 0, 2))
    cv2.attach_upper_wall(location=3.14, kappa=10000)

    # Attach CVs to a bias and initialise it for metadynamics AL.

    # By default, metadynamics bias is stored as a list of deposited 
    # gaussians, which results in every MD step scaling linearly with 
    # the total length of the simulation. To make the scaling constant, 
    # the bias can be stored on a grid. This requires to specify the 
    # bounds for the grid, and the bounds should be chosen such that 
    # during AL the system would not leave the grid (either by using 
    # a large grid or attaching walls to constrain the system).

    # Other metadynamics parameters can also be set by the method,
    # see initialise_for_metad_al() documentation for more info.

    bias = mlt.PlumedBias(cvs=(cv1, cv2))
    bias.initialise_for_metad_al(width=(0.05, 0.10), biasfactor=100)

    # Define the potential and train it using metadynamics inherited-bias AL.
    # Inheritance can be set to False, the same initialisation works.
    # Metadynamics bias starts being applied at iteration 2, at iterations 0
    # and 1 the training is performed using unbiased MD with the attached walls

    ace = mlt.potentials.ACE('water', system=h2o_system)
    ace.al_train(method_name='xtb',
                 temp=300,
                 max_active_iters=50,
                 min_active_iters=5,
                 bias_start_iter=2, 
                 inherit_metad_bias=True,
                 bias=bias)

    # NOTE: The same al_train() method works with arbitrary PLUMED biases 
    # (i.e. not only metadynamics) by initialising a PlumedBias using a 
    # PLUMED input file, but then inheritance is unavailable
