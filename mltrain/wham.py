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
                  n_steps=1000,
                  r0=0.5,
                  dt=0.1
                  ) -> np.ndarray:
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
        arr:
    """
    def a(_r, m=1.0):
        """F = m a  --> a = -∇(U + w) / m  where m is the mass"""
        return - (u.grad(_r) + (0.0 if w is None else w.grad(_r))) / m

    r_mt, r_t = r0, r0     # r(t - δt), r(t)

    qs = []

    for _ in range(n_steps):

        r = 2*r_t - r_mt + a(r_t) * dt**2
        qs.append(f(r))

        r_mt, r_t = r_t, r

    return np.array(qs)


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

        self.w = W(f=f, s=s, kappa=100)   # Bias potential
        self.A = 0.0                      # Estimate of the free energy

        self.bins = np.linspace(np.min(q), np.max(q), num=len(q) + 1)
        self.h = None                     # Histogram
        self.W = self.w(q)                # Value of the bias

    @property
    def n(self) -> int:
        """Number of samples in this window"""
        return int(np.sum(self.h))

    def sample(self) -> None:
        """Sample this window, populating the histogram of q values"""
        self.h, _ = np.histogram(verlet_sample(self.u, w=self.w, r0=self.w.s),
                                 bins=self.bins)

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

        for window in self.windows:
            window.sample()

        return None

    def wham(self,
             beta=1.0,
             tol=1E-8,
             max_iterations=1000
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
        p = np.exp(-(self.q - 1.0)**2)          # and P(q)

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

            p_prev = p

        self.p = p
        return None


if __name__ == '__main__':

    # u = U()
    # w = W(f=LinearTransform(), s=0.3, kappa=1)
    # plt.hist(verlet_sample(u, w), density=True, bins=10)
    # plot(u, w)

    umbrella = Umbrella(r_min=-0.2, r_max=2.0, n_windows=40, n_bins=20)
    umbrella.sample()
    umbrella.wham()
    print(umbrella.A)
