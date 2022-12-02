import mlptrain as mlt
import autode as ade

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ade.Config.ORCA.keywords.grad


if __name__ == '__main__':

    system = mlt.System(mlt.Molecule('ts.xyz'), box=None)

    ace = mlt.potentials.ACE('da', system=system)

    ace.al_train_then_bias(method_name='orca',
                           coordinate=mlt.AverageDistance((0, 1), (2, 3)),
                           max_coordinate=3.5,
                           selection_method=mlt.training.selection.AbsDiffE(0.043),
                           temp=500,
                           max_active_time=500,
                           fix_init_config=True)

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(configuration=system.configuration,
                                   mlp=ace,
                                   fs=500,
                                   temp=300,
                                   dt=0.5,
                                   interval=10)

    # and compare, plotting a parity diagram and E_true, ∆E and ∆F
    trajectory.compare(ace, 'orca')

