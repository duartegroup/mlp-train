import os
import numpy as np
from mlptrain.sampling.bias import Bias
from mlptrain.sampling.umbrella import _Window, UmbrellaSampling
from .utils import work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))

kj_to_ev = 0.0103642


def _initialised_us() -> UmbrellaSampling:
    us = UmbrellaSampling(zeta_func=lambda x: None, kappa=0.0)

    us.temp = 300
    zeta_refs = np.linspace(1.8245, 3.1100, num=20)  # 20 windows

    for window_idx in range(20):
        data_lines = open(f'window_{window_idx+1}.txt', 'r').readlines()

        # Ensure the data has the correct reference value for the hard
        # coded array
        assert np.isclose(
            zeta_refs[window_idx], float(data_lines[0].split()[1])
        )

        zeta_obs = [float(line.split()[1]) for line in data_lines[1:-1]]

        window = _Window(
            obs_zetas=np.array(zeta_obs),
            bias=Bias(
                zeta_func=None,
                kappa=float(data_lines[0].split()[2]),
                reference=zeta_refs[window_idx],
            ),
        )

        us.windows.append(window)

    return us


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_wham_is_close_to_ref():
    us = _initialised_us()
    zetas, free_energies = us.wham(n_bins=499)
    free_energies -= min(free_energies)

    ref_zetas = np.array(
        [
            float(line.split()[0])
            for line in open('ref_wham.txt', 'r').readlines()[1:-1]
        ]
    )

    ref_free_energies = [
        float(line.split()[1]) * kj_to_ev
        for line in open('ref_wham.txt', 'r').readlines()[1:-1]
    ]

    # Ensure every free energy value, at a particular zeta, is close to the
    # reference, to within ~0.5 kcal mol-1
    for zeta, free_energy in zip(zetas, free_energies):
        close_idx = np.argmin(np.abs(ref_zetas - zeta))
        assert np.isclose(free_energy, ref_free_energies[close_idx], atol=0.02)


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_wham_is_somewhat_independent_of_nbins():
    us = _initialised_us()
    _, free_energies_500bins = us.wham(n_bins=500)
    _, free_energies_1000bins = us.wham(n_bins=1000)

    assert np.allclose(
        free_energies_500bins,
        np.mean(
            free_energies_1000bins.reshape(-1, 2), axis=1
        ),  # block average
        atol=5e-2,
    )
