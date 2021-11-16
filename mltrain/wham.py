"""
Umbrella sampling with WHAM unbiasing

References:
[1] https://pubs.acs.org/doi/10.1021/ct501130r
[2] https://pubs.acs.org/doi/pdf/10.1021/ct100494z
"""
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Function(ABC):

    @abstractmethod
    def __call__(self, r):
        """Value of the potential"""

    @abstractmethod
    def grad(self, r):
        """Gradient of the potential dU/dr"""


class U(Function):
    """Unbiased potential"""

    def __call__(self, r):
        return r**2 * (r - 1.5)**2

    def grad(self, r):
        return 2*r * (r - 1.5)**2 + r**2 * 2*(r - 1.5)


class W(Function):
    """Biasing potential"""

    def __init__(self,
                 f:     Function,
                 s:     float,
                 kappa: float = 1.0):
        """
        Biasing potential:    W = κ/2 (f(r) - s)^2

        -----------------------------------------------------------------------
        Arguments:
            f: Function to transform the coordinate

            s: Reference value of the bias

        Keyword Arguments:
            kappa: Strength of the biasing potential
        """

        self.f = f
        self.s = s
        self.kappa = kappa

    def __call__(self, r):
        """Value of the bias"""
        return 0.5 * self.kappa * (self.f(r) - self.s)**2

    def grad(self, r):
        """Gradient of the biasing potential"""
        return self.kappa * self.f.grad(r) * (self.f(r) - self.s)


class LinearTransform(Function):
    """Linear transform of the coordinate r -> r"""

    def __call__(self, r):
        return r

    def grad(self, r):
        return 1.0


def plot(*args, r_min=-0.7, r_max=2.2):
    """Plot a function over a range of rs"""

    rs = np.linspace(r_min, r_max, num=500)
    for function in args:
        plt.plot(rs, function(rs))
        plt.plot(rs, function.grad(rs), ls='--')

    # plt.ylim(-1, 2)
    plt.xlabel('$r$')
    plt.tight_layout()
    plt.savefig(f'plot.pdf')

    return None


def verlet_sample(u,
                  w=None,
                  f=LinearTransform(),
                  n_steps=10000,
                  r0=0.5,
                  dt=0.1
                  ) -> dict:
    """
    Sample a potential using a Verlet algorithm (NVE sampling)

    --------------------------------------------------------------------------
    Arguments:
        u: Potential

        w: Biasing potential

        f: Transform function q = f(r)

        n_steps: Number of steps to perform

        r0: Starting position

        dt: Time step

    Returns:
        dict(str, list):
    """
    def a(_r, m=1.0):
        """F = m a  --> a = -∇(U + w) / m  where m is the mass"""
        return - (u.grad(_r) + (0.0 if w is None else w.grad(_r))) / m

    r_mt, r_t = r0, r0     # r(t - δt), r(t)

    data = {'qs': [], 'us': []}

    for _ in range(n_steps):

        r = 2*r_t - r_mt + a(r_t) * dt**2
        data['qs'].append(f(r))
        data['us'].append(u(r))

        r_mt, r_t = r_t, r

    return data


class Window:

    def __init__(self,
                 s: float,
                 u: Function,
                 f: Function,
                 q: np.ndarray):
        """
        Umbrella Window

        -----------------------------------------------------------------------
        Arguments:
            s: Reference value of the bias

            u: Unbiased potential to sample

            f: Transform function r -> q
        """
        self.u = u

        self.w = W(f=f, s=s, kappa=5)   # Bias potential
        self.A = 0.0                      # Estimate of the free energy

        self.bins = np.linspace(np.min(q), np.max(q), num=len(q) + 1)
        self.h = None                     # Histogram
        self.W = self.w(q)                # Value of the bias

    @property
    def n(self) -> int:
        """Number of samples in this window"""
        return int(np.sum(self.h))

    def sample(self, name=None) -> None:
        """Sample this window, populating the histogram of q values"""
        data = verlet_sample(self.u, w=self.w, r0=self.w.s)
        self.h, _ = np.histogram(np.array(data['qs']), bins=self.bins)

        if name is not None:

            with open(f'{name}.txt', 'w') as data_file:
                for idx, q in enumerate(data['qs']):
                    print(idx, q, 0.0, file=data_file)

        return None


class Umbrella:
    """Umbrella sampling"""

    def __init__(self,
                 r_min:     float,
                 r_max:     float,
                 n_windows: int,
                 n_bins:    int,
                 u:         Function = U(),
                 f:         Function = LinearTransform()):
        """
        Collection of umbrella windows, constructed on a potential using a
        bias over which WHAM can be performed

        -----------------------------------------------------------------------
        Arguments:

            r_min: Minimum value of r to sample

            r_max: Maximum value of r to sample

            n_windows: Number of windows to use

            n_bins: Number of bins to use in the WHAM

        Keyword Arguments:
            u: Potential to use

            f: Transform of the coordinates (r) to q (scalar)
        """
        self.q = np.linspace(f(r_min), f(r_max), num=n_bins)

        # Probability distribution
        self.p = None

        s = np.linspace(min(self.q), max(self.q), num=n_windows)
        self.windows = [Window(s=s_k, u=u, f=f, q=self.q) for s_k in s]

    @property
    def A(self) -> np.ndarray:
        """Free energy estimates"""
        return np.array([w_k.A for w_k in self.windows])

    def sample(self) -> None:
        """Sample all the windows in this set"""

        for i, window in enumerate(self.windows):
            window.sample(name=f'window_{i}')

        with open('metadata.txt', 'w') as meta_file:
            for i, window in enumerate(self.windows):
                print(f'window_{i}.txt', window.w.s, window.w.kappa, 10,
                      file=meta_file)

        return None

    def wham(self,
             beta=1.0,
             tol=1E-3,
             max_iterations=10000
             ) -> None:
        """
        Construct an unbiased distribution (on a grid) from a set of windows

        -----------------------------------------------------------------------
        Arguments:
            beta: 1 / k_B T

            tol: Tolerance on the convergence

            max_iterations: Maximum number of WHAM iterations to perform
        """
        p_prev = np.inf * np.ones_like(self.q)  # Start with P(q)^(-1) = ∞
        p = np.ones_like(self.q) / len(self.q)  # and P(q) as a uniform distro.

        def converged():
            return np.max(np.abs(p_prev - p)) < tol

        for _ in range(max_iterations):

            p = (sum(w_k.h for w_k in self.windows)
                 / sum(w_k.n * np.exp(beta * (w_k.A - w_k.W))
                       for w_k in self.windows))

            for w_k in self.windows:
                w_k.A = (-(1.0/beta)
                         * np.log(np.sum(p * np.exp(-w_k.W * beta))))

            if converged():
                break

            # print(np.max(np.abs(p_prev - p)))
            p_prev = p

        self.p = p
        return None


def parse_free_energy_txt():

    read = False
    es = []

    for line in open('free_energy.txt', 'r'):

        if read and len(line.split()) == 0:
            break

        if read:
            es.append(float(line.split()[1]))

        if 'Window' in line:
            read = True

    return np.array(es)


def plot1d_example():
    umbrella = Umbrella(r_min=-0.1, r_max=0.8, n_windows=4, n_bins=50)
    umbrella.sample()
    umbrella.wham()
    print(umbrella.A - umbrella.A[0])

    import matplotlib.pyplot as plt
    plt.style.use('paper')
    plt.plot(parse_free_energy_txt(), label='WHAM')
    plt.plot(umbrella.A - umbrella.A[0], label='me')

    plt.legend()
    plt.tight_layout()
    plt.savefig('tmp.pdf')


if __name__ == '__main__':

    # u = U()
    # w = W(f=LinearTransform(), s=0.3, kappa=1)
    # plt.hist(verlet_sample(u, w), density=True, bins=10)
    # plot(u, w)

    umbrella = Umbrella(r_min=1.4, r_max=3.7, n_windows=30, n_bins=50)
    metadata = open('metadata.txt', 'r').readlines()
    for idx, window in enumerate(umbrella.windows):
        window.w.s = float(metadata[idx].split()[1])
        window.w.kappa = float(metadata[idx].split()[2])

        qs = [float(line.split()[1])
              for line in open(f'window_{idx}.txt').readlines()[1:]]

        window.W = window.w(umbrella.q)

        window.h, _ = np.histogram(np.array(qs), bins=window.bins)

    umbrella.wham(beta=(1/0.1))

    plt.plot(umbrella.A - min(umbrella.A))
    wham_As = parse_free_energy_txt()
    plt.plot(wham_As - min(wham_As))
    plt.tight_layout()
    plt.savefig('tmp.pdf')
