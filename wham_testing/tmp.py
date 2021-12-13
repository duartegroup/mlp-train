import numpy as np
import matplotlib.pyplot as plt
from mltrain.sampling.reaction_coord import AverageDistance
from mltrain.sampling.umbrella import (_Window,
                                       UmbrellaSampling)
plt.style.use('paper')


if __name__ == '__main__':

    us = UmbrellaSampling(zeta_func=AverageDistance(),
                          kappa=0)
    us.temp = 300
    us.n_points = 500

    refs = np.linspace(1.8245029136452484,
                       3.11,
                       num=20)
    us.refs = refs
    q_points = np.linspace(refs[0], refs[-1], num=us.n_points)

    for i in range(20):
        zetas = [float(line.split()[1])
                 for line in open(f'window_{i}.txt', 'r').readlines()[1:-1]]
        zetas = np.array(zetas)

        ref_zeta = float(open(f'window_{i}.txt', 'r').readline().split()[1])
        kappa = float(open(f'window_{i}.txt', 'r').readline().split()[2])

        us.windows.append(_Window(rxn_coords=zetas,
                                  bias_e=0.5 * kappa * (q_points - ref_zeta)**2,
                                  refs=refs,
                                  n_points=us.n_points))


    us.wham()
    plt.close()

    xs = [float(line.split()[0]) for line in open('wham.out', 'r').readlines()[1:-1]]
    ys = [float(line.split()[1]) for line in open('wham.out', 'r').readlines()[1:-1]]

    rel_free_energies = us.free_energies - min(us.free_energies)

    plt.plot(q_points, 23.0605419*rel_free_energies, color='k', label='us')
    plt.plot(xs, 0.2390*np.array(ys), color='b', label='rust')

    plt.legend()
    plt.tight_layout()
    plt.savefig('tmp.pdf')
