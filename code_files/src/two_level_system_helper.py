from .package_requirements import *

# basis states
ket0 = qt.basis(2, 0)
ket1 = qt.basis(2, 1)


def collision_hamiltonian(omega: float, kappa:float) -> qt.Qobj:
    return omega * (qt.tensor(qt.qeye(2), qt.sigmax())) + kappa * (
            qt.tensor(qt.sigmam(), qt.sigmap()) + qt.tensor(qt.sigmap(), qt.sigmam()))


def collision_unitary(hamiltonian: qt.Qobj) -> qt.Qobj:
    return (-1j * hamiltonian).expm()
