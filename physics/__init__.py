from .periodicbc import simulate_periodic_ring
from .openbc   import simulate_open_chain
from .isingopen import simulate_ising_open_chain

REGISTRY = {
    "periodic": simulate_periodic_ring,
    "open": simulate_open_chain,
    "ising_open": simulate_ising_open_chain,
}
