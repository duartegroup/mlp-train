from mltrain.sampling._base import Function, ASEConstraint


class Bias(ASEConstraint, Function):
    """Modifies the forces and energy of a set of ASE atoms under a bias"""

    def __init__(self,
                 zeta_func: 'mltrain.sampling.reaction_coord.ReactionCoordinate',
                 kappa:     float,
                 reference: float):
        """
        Bias that modifies the forces and energy of a set of atoms under a
        harmonic bias function.

        Harmonic biasing potential: ω = κ/2 (ζ(r) - ζ_ref)^2

        e.g. bias = mlt.bias.Bias(to_average=[[0, 1]], reference=2, kappa=10)

        -----------------------------------------------------------------------
        Arguments:

            zeta_func: Reaction coordinate, taking the positions of the system
                     and returning a scalar e.g. a distance or sum of distances

            kappa: Value of the spring constant, κ

            reference: Reference value of the reaction coordinate, ζ_ref
        """
        self.ref = reference
        self.kappa = kappa
        self.f = zeta_func

    def __call__(self, atoms):
        """Value of the bias for set of atom pairs in atoms"""

        return 0.5 * self.kappa * (self.f(atoms) - self.ref)**2

    def grad(self, atoms):
        """Gradient of the biasing potential a set of atom pairs in atoms"""

        return self.kappa * self.f.grad(atoms) * (self.f(atoms) - self.ref)

    def adjust_potential_energy(self, atoms):
        """Adjust the energy of a set of atoms using the bias function"""
        return self.__call__(atoms)

    def adjust_forces(self, atoms, forces):
        """Adjust the forces of a set of atoms in place using the gradient
        of the bias function::

         F = -∇E -∇B

        where ∇E is the gradient of the energy with respect to the coordinates
        and B is the bias.
        """
        forces -= self.grad(atoms)
        return None

    def adjust_positions(self, atoms, newpositions):
        """Method required for ASE but not used in ml-train"""
        return None
