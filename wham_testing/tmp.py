import numpy as np
import matplotlib.pyplot as plt
from mltrain.sampling.reaction_coord import AverageDistance
from mltrain.sampling.bias import Bias
from mltrain.sampling.umbrella import (_Window,
                                       UmbrellaSampling)
plt.style.use('paper')


if __name__ == '__main__':

    us = UmbrellaSampling(zeta_func=AverageDistance(),
                          kappa=0)
    us.temp = 300

    refs = np.linspace(1.8245029136452484,
                       3.11,
                       num=20)
    us.zeta_refs = refs

    for i in range(20):
        zetas = [float(line.split()[1])
                 for line in open(f'window_{i}.txt', 'r').readlines()[1:-1]]
        zetas = np.array(zetas)

        ref_zeta = float(open(f'window_{i}.txt', 'r').readline().split()[1])
        kappa = float(open(f'window_{i}.txt', 'r').readline().split()[2])

        us.windows.append(_Window(obs_zetas=zetas,
                                  bias=Bias(zeta_func=None,
                                            kappa=kappa,
                                            reference=ref_zeta)))

    zetas, A = us.wham(n_bins=300)
    plt.close()

    xs = [float(line.split()[0]) for line in open('wham.out', 'r').readlines()[1:-1]]
    ys = [float(line.split()[1]) for line in open('wham.out', 'r').readlines()[1:-1]]

    plt.plot(zetas, 23.0605419*(A - min(A)), color='k', label='us')
    plt.plot(xs, 0.2390*np.array(ys), color='b', label='rust')

    plt.legend()
    plt.tight_layout()
    plt.savefig('tmp2.pdf')
