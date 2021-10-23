import mltrain as mlt

if __name__ == '__main__':

    methane = mlt.Molecule('methane.xyz')

    system = mlt.System(box=[10, 10, 10])
    system.add_molecules(methane, num=12)

    system.random_configuration().save('tmp.xyz')
