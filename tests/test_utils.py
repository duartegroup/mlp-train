import mlptrain as mlt
from mlptrain import utils
import os


@utils.work_in_tmp_dir()
def save_npz_for_test(
    npz_filename: str = 'filename.npz',
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


def test_save_duplicate_npz():
    # Save empty file as test.xyz
    with open('empty_test.xyz', 'w'):
        pass

    save_npz_for_test(npz_filename='empty_test.npz')

    assert (
        utils.npz_to_xyz() is None
    )  # Should return None without error when trying to overwrite existing .xyz file

    assert os.path.exists('empty_test.xyz')  # xyz file should still exist

    assert (
        os.path.getsize('empty_test.npz') == 0
    )  # Previous empty npz file should not have been overwritten


def test_save_nonexistent_npz():
    # Ensure that trying to convert a non-existent npz file raises FileNotFoundError
    assert not utils.npz_to_xyz('nonexistent_file.npz')


def test_npz_to_xyz_conversion():
    save_npz_for_test(npz_filename='convert_test.npz')

    utils.npz_to_xyz('convert_test.npz')

    assert os.path.exists(
        'convert_test.xyz'
    )  # Check that the .xyz file was created

    assert mlt.ConfigurationSet().load(
        'convert_test.xyz'
    )  # Check that the .xyz file can be loaded without error
