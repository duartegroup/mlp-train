import mlptrain as mlt
from mlptrain import utils
import os
from autode.atoms import Atom
import pytest


def save_npz_for_test(
    npz_filename: str,
):
    """
    Take a filename as input.
    Save a ConfigurationSet (containing 2 Configurations) as an npz file under that name.
    Return the path to the directory the file is saved in.
    Should only be called within a tmp directory while testing.
    """
    configset = mlt.ConfigurationSet()

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

    configset.load_xyz(
        'tmp.xyz', charge=0, mult=1, load_energies=True, load_forces=True
    )
    configset.save(npz_filename)

    os.remove('tmp.xyz')

    return os.getcwd()


@utils.work_in_tmp_dir()
def test_npz_to_xyz_missing_extension():
    """Ensure that calling npz_to_xyz(filename) raises ValueError if filename doesn't end with .npz."""

    save_npz_for_test(npz_filename='convert_test.npz')

    with pytest.raises(ValueError):
        utils.npz_to_xyz('convert_test')


@utils.work_in_tmp_dir()
def test_save_nonexistent_npz():
    """Ensure that trying to convert a non-existent npz file raises FileNotFoundError"""

    with pytest.raises(FileNotFoundError):
        utils.npz_to_xyz('nonexistent_file.npz')


@utils.work_in_tmp_dir()
def test_save_duplicate_xyz():
    """
    Create an xyz file first, then create an npz file with the same name (excluding extension).
    Ensure that trying to convert the npz file to an xyz file already exists raises a RuntimeError rather than overwriting the existing file.
    """
    atoms = [
        Atom('H', 0.0, 0.0, 0.0),
        Atom('H', 1.0, 0.0, 0.0),
        Atom('O', 0.5, 0.5, 0.0),
    ]

    my_config = mlt.Configuration(atoms=atoms, charge=0, mult=1)
    orig_configset = mlt.ConfigurationSet(my_config)
    orig_configset.save('duplicate_test.xyz')

    # Create an npz file of a different ConfigurationSet, with the same name [excluding extension]
    save_npz_for_test(npz_filename='duplicate_test.npz')

    with pytest.raises(RuntimeError):
        utils.npz_to_xyz(npz_filename='duplicate_test.npz')

    assert os.path.exists('duplicate_test.xyz')

    current_configset = mlt.ConfigurationSet()
    current_configset.load_xyz('duplicate_test.xyz', charge=0, mult=1)

    assert isinstance(current_configset, mlt.ConfigurationSet)

    assert len(current_configset) == 1
    assert len(current_configset[0].atoms) == 3

    assert 'C' not in [atom.label for atom in current_configset[0].atoms]


@utils.work_in_tmp_dir()
def test_npz_to_xyz_conversion():
    """Test that converting a regular npz file to xyz works correctly."""

    path = save_npz_for_test(npz_filename='convert_test.npz')

    utils.npz_to_xyz(f'{path}/convert_test.npz')

    assert os.path.exists(f'{path}/convert_test.xyz')

    configset = mlt.ConfigurationSet()
    configset.load_xyz(f'{path}/convert_test.xyz', charge=0, mult=1)

    assert isinstance(configset, mlt.ConfigurationSet)

    assert len(configset) == 2

    assert len(configset[0].atoms) == 3
    assert len(configset[1].atoms) == 2
