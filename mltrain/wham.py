"""
Umbrella sampling with WHAM unbiasing

References:
[1] https://pubs.acs.org/doi/10.1021/ct501130r
[2] https://pubs.acs.org/doi/pdf/10.1021/ct100494z
[3] https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.66
"""
import matplotlib.pyplot as plt
plt.style.use('paper')


import numpy as np
from typing import Optional
from scipy.optimize import curve_fit
from scipy.integrate import simpson
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


class NormalPDF:

    def __init__(self,
                 mu:    float,
                 sigma: float):
        """Normal probability density function (a Gaussian)"""

        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return ((1.0 / (self.sigma * np.sqrt(2.0 * np.pi)))
                * np.exp(-0.5 * ((x - self.mu) / self.sigma)**2))


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
                  dt=0.1,
                  v0=0.01) -> dict:
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

    r_mt = r0 - v0 * dt + 0.5*a(r0) * dt**2
    r_t = r0     # r(t - δt), r(t)

    data = {'qs': [], 'us': []}

    for _ in range(n_steps):

        r = 2*r_t - r_mt + a(r_t) * dt**2
        data['qs'].append(f(r))
        data['us'].append(u(r))

        r_mt, r_t = r_t, r

    return data


def velocity_verlet_berendsen(u,
                              w=None,
                              f=LinearTransform(),
                              n_steps=10000,
                              r0=0.5,
                              dt=0.1,
                              v0=0.01,
                              temp=0.0001
                              ) -> dict:
    """
    Perform sampling with the velocity verlet algorithm, while rescaling
    the velocity to maintain a ~NVT ensemble
    """
    def a(_r, m=1.0):
        return - (u.grad(_r) + (0.0 if w is None else w.grad(_r))) / m

    def v_factor(m=1.0):
        return 1.0 + 0.5 * (temp / (m * v_t**2) - 1)

    r_t = r0   # r(t )
    v_t, v_pt = np.sqrt(temp), v0

    # KE = 0.5 * k * T = 0.5 * m * v**2
    # v = sqrt(kT/m)    where k = 1

    data = {'qs': [], 'us': []}

    for _ in range(n_steps):

        data['qs'].append(f(r_t))
        data['us'].append(u(r_t))

        v_t *= v_factor()
        r_pt = r_t + v_t * dt + 0.5 * a(r_t) * dt**2
        v_pt = v_t + 0.5 * (a(r_t) + a(r_pt)) * dt

        r_t, v_t = r_pt, v_pt

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

            q: Values of the reaction coordinate (q)
        """
        self.u = u                        # Unbiased potential

        self.w = W(f=f, s=s, kappa=15)    # Bias potential
        self.A = 0.0                      # Estimate of the free energy

        self.q = q                        # Values of the reaction coordinate

        self._sampled_qs: Optional[np.ndarray] = None
        self.h: Optional[np.ndarray] = None         # Histogram of q values
        self.W = self.w(q)                          # Values of the bias

        self._pdf: Optional[NormalPDF] = None       # Probability density func.

    @property
    def n(self) -> int:
        """Number of samples in this window"""
        return int(np.sum(self.h))

    @property
    def bins(self) -> np.ndarray:
        """Bin edges to histogram into"""
        return np.linspace(np.min(self.q), np.max(self.q), num=len(self.q) + 1)

    @property
    def pdf(self) -> NormalPDF:
        """Probability density function"""

        if self._pdf is None:
            self._fit_normal_distro()

        return self._pdf

    @property
    def more_q(self) -> np.ndarray:
        """Return some more q values in the same range"""
        return np.linspace(np.min(self.q), np.max(self.q), num=500)

    def dAu_dq(self, q, beta):
        """Equation 20 from ref """
        return ((1.0 / beta) * (q - self.pdf.mu) / (self.pdf.sigma**2)
                - self.w.grad(q))

    def sample(self, name=None) -> None:
        """Sample this window, populating the histogram of q values"""
        data = velocity_verlet_berendsen(self.u, w=self.w, r0=self.w.s)
        self._sampled_qs = np.array(data['qs'])

        self.h, _ = np.histogram(self._sampled_qs, bins=self.bins)

        if name is not None:

            with open(f'{name}.txt', 'w') as data_file:
                for idx, q in enumerate(data['qs']):
                    print(idx, q, 0.0, file=data_file)

        return None

    def _fit_normal_distro(self) -> None:
        """
        Fit a normal PDF to the data
        """
        if self.h is None:
            raise ValueError(f'Cannot fit a normal PDF - no data')

        def func(_q, a, b, c):
            return a * NormalPDF(b, c)(_q)

        a_0, mu_0, sigma_0 = (np.max(self.h),
                              np.average(self._sampled_qs),
                              float(np.std(self._sampled_qs)))

        try:
            opt, _ = curve_fit(func, self.q, self.h, p0=[a_0, 0.0, 0.2])
            self._pdf = NormalPDF(mu=opt[1], sigma=opt[2])

        except RuntimeError:
            self._pdf = NormalPDF(mu=mu_0, sigma=sigma_0)

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
        self.A = np.zeros_like(self.q)    # Free energy at each coordinate

        # Uniform probability distribution starting point
        self.p = np.ones_like(self.q) / len(self.q)

        s = np.linspace(min(self.q), max(self.q), num=n_windows)
        self.windows = [Window(s=s_k, u=u, f=f, q=self.q) for s_k in s]

    @property
    def A_centres(self) -> np.ndarray:
        """
        Free energy estimates for each of the windows

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray):
        """
        free_energies = np.array([w_k.A for w_k in self.windows])
        return free_energies - free_energies[0]

    @property
    def q_centres(self) -> np.ndarray:
        """
        Reference values of the reaction coordinate for each of the windows

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray):
        """
        return np.array([w_k.w.s for w_k in self.windows])

    @property
    def dq(self) -> float:
        """Spacing in the array of reaction coordinates (q). Expecting
        the array to be evenly spaced

        -----------------------------------------------------------------------
        Returns:
            (float):
        """
        if len(self.q) < 2:
            raise ValueError('Cannot determine dq with fewer than 2 samples')

        return self.q[1] - self.q[0]

    def sample(self) -> None:
        """Sample all the windows in this set"""

        for i, window in enumerate(self.windows):
            window.sample() # name=f'window_{i}')

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
        p = self.p

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

    def ui(self,
           beta=1.0) -> None:
        """
        Perform umbrella integration on the umbrella windows to un-bias the
        probability distribution. Such that the the PMF becomes

        .. math::
            dA/dq = Σ_i p_i(q) dA^u_i/ dq

        where the sum runs over the windows.
        """
        q_arr = np.linspace(np.min(self.q), np.max(self.q), num=len(self.q))

        dA_dq = np.zeros_like(self.q)
        sum_a = 0

        for i, window in enumerate(self.windows):

            a_i = window.n * window.pdf(q_arr)
            dA_dq += a_i * window.dAu_dq(q_arr, beta=beta)
            sum_a += a_i

        # Normalise
        dA_dq /= sum_a

        for i, q_val in enumerate(self.q):
            if i == 0:
                self.A[i] = 0.0

            else:
                self.A[i] = simpson(dA_dq[:i],
                                    self.q[:i],
                                    dx=self.dq)

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
    umbrella = Umbrella(r_min=-0.2, r_max=0.2, n_windows=10, n_bins=100)
    umbrella.sample()
    umbrella.wham(beta=1/0.001)

    # plt.plot(parse_free_energy_txt(), label='WHAM')
    plt.plot(umbrella.q_centres, umbrella.A_centres, label='WHAM')
    for window in umbrella.windows[1:]:
        plt.bar(umbrella.q,
                window.h/np.max(window.h)*0.2,
                width=(umbrella.q[1] - umbrella.q[0]),
                alpha=0.2,
                edgecolor='k')

        pdf = window.pdf(window.more_q)
        print(window.pdf.mu)
        plt.plot(window.more_q,
                 pdf / np.max(pdf) * 0.2,
                 )

    if True:
        umbrella.ui(beta=1/0.001)
        print(umbrella.A)
        plt.plot(umbrella.q, umbrella.A, label='UI', c='orange')

    plt.legend()
    plt.tight_layout()
    plt.savefig('tmp.pdf')


if __name__ == '__main__':

    # u = U()
    # w = W(f=LinearTransform(), s=0.3, kappa=1)
    # plt.hist(verlet_sample(u, w), density=True, bins=10)
    # plot(u, w)

    """
    tmp = Window(0.1, u=U(), f=LinearTransform(),
                 q=np.linspace(-0.1, 1.5, num=100))
    tmp.sample()

    plt.plot(tmp.more_q,
             tmp.pdf(tmp.more_q))
    plt.savefig('tmp.pdf')
    """

    plot1d_example()
