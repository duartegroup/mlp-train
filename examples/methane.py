import mltrain as mlt

if __name__ == '__main__':

    system = mlt.System(mlt.Molecule('methane.xyz'),
                        box=[50, 50, 50])   # Å

    system.random_configuration().save('tmp.xyz')
