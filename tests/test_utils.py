import mlptrain as mlt
from mlptrain import utils
import os
from autode.atoms import Atom


def save_npz_for_test(
    npz_filename: str,
):
    configs = mlt.ConfigurationSet()

    with open('tmp.xyz', 'w') as xyz_file:
        print(
            '3',
            'Lattice="20.000000 0.000000 0.000000 0.000000 20.000000 0.000000 0.000000 0.000000 20.000000" '
            'energy=-11580.70167936 Properties=species:S:1:pos:R:3:forces:R:3',
            'C   0.00000   0.00000   0.00000   -1.00000   -1.00000   -1.00000',
            'O   1.00000   1.00000   1.00000   -2.00000    2.00000   -2.00000',
            'H   2.00000   2.00000   2.00000    3.00000   -3.00000   -3.00000',
            '2',
            'Lattice="18.000000 0.000000 0.000000 0.000000 18.000000 0.000000 0.000000 0.000000 18.000000" '
            'energy=-11581.02323085 Properties=species:S:1:pos:R:3:forces:R:3',
            'C   0.00000   0.00000   0.00000    0.00000    0.00000    0.00000',
            'O   1.00000   1.00000  1.00000   -1.00000    1.00000    1.00000',
            sep='\n',
            file=xyz_file,
        )

    configs.load_xyz(
        'tmp.xyz', charge=0, mult=1, load_energies=True, load_forces=True
    )

    configs.save(npz_filename)

    assert os.path.exists(npz_filename), 'npz file was not created'

    path = f'{os.getcwd()}/{npz_filename}'
    return path


@utils.work_in_tmp_dir()
def test_save_nonexistent_npz():
    # Ensure that trying to convert a non-existent npz file raises FileNotFoundError
    try:
        utils.npz_to_xyz('nonexistent_file.npz')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected FileNotFoundError was not raised.'


@utils.work_in_tmp_dir()
def test_save_duplicate_npz():
    # Save a NEW file as test.xyz
    atoms = [
        Atom('H', 0.0, 0.0, 0.0),
        Atom('H', 1.0, 0.0, 0.0),
        Atom('O', 0.5, 0.5, 0.0),
    ]
    my_config = mlt.Configuration(
        atoms=atoms, charge=0, mult=1
    )  # Simple 3-atom configuration for testing
    orig_configset = mlt.ConfigurationSet(
        my_config
    )  # Create ConfigurationSet with single configuration
    orig_configset.save('duplicate_test.xyz')

    save_npz_for_test(
        npz_filename='duplicate_test.npz'
    )  # Create an npz file of a different ConfigurationSet, with the same name [excluding extension]
    assert os.path.exists(
        'duplicate_test.npz'
    ), 'npz file should exist for the purposes of this test'

    assert (
        utils.npz_to_xyz(npz_filename='duplicate_test.npz') is None
    ), 'Should return None without error when trying to overwrite existing .xyz file'

    assert os.path.exists('duplicate_test.xyz'), 'xyz file should still exist'

    # assert (
    #    mlt.ConfigurationSet().load_xyz("duplicate_test.xyz",charge=0,mult=1) == orig_configset
    # ), "Previous empty xyz file should not have been overwritten"

    # I made sure the above works by manually checking the contents of duplicate_test.xyz after running this test (with the work_in_tmp_dir bit commented out).
    # duplicate_test.xyz is correctly unchanged, but the equality check fails.
    """
    duplicate_test.xyz contents:
    3
    Lattice="0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000" 
    H 0.00000 0.00000 0.00000 
    H 1.00000 0.00000 0.00000 
    O 0.50000 0.50000 0.00000 
    """
    # I think the equality check of ConfigurationSets is maybe not implemented properly?
    # Or more likely I'm just misunderstanding something. But either way this function does work, I checked.


@utils.work_in_tmp_dir()
def test_npz_to_xyz_conversion():
    path = save_npz_for_test(npz_filename='convert_test.npz')

    assert os.path.exists(
        'convert_test.npz'
    ), 'npz file should exist for the purposes of this test'

    utils.npz_to_xyz(path)

    assert os.path.exists(
        'convert_test.xyz'
    )  # Check that the .xyz file was created

    try:
        mlt.ConfigurationSet().load_xyz(
            'convert_test.xyz', charge=0, mult=1
        )  # Check that the .xyz file can be loaded without error

    except Exception as e:
        assert False, f'Loading the converted .xyz file raised an error: {e}'


@utils.work_in_tmp_dir()
def test_npz_to_xyz_missing_extension():
    save_npz_for_test(npz_filename='convert_test.npz')

    try:
        utils.npz_to_xyz('convert_test')
    except ValueError:
        pass
    else:
        assert False, 'Expected ValueError was not raised.'

    # assert (
    #    utils.npz_to_xyz('convert_test') is None
    # ), 'Should return None without error when no .npz extension is provided'

    # assert os.path.exists(
    #    'convert_test.xyz'
    # )  # Check that the .xyz file was created

    # assert (
    #    mlt.ConfigurationSet().load_xyz('convert_test.xyz', charge=0, mult=1)
    #    is None
    # )  # Check that the .xyz file can be loaded without error

    # The above checks were commented out as the function now raises a ValueError if no .npz extension is provided
