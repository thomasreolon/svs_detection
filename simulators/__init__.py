from .forensor_sim import StaticSVS
from .grey import GreyscaleSVS
from .rlearn import RLearnSVS

supported = [
    StaticSVS,      # static
    GreyscaleSVS,    # grey
    RLearnSVS
]

def get_simulator(name):
    for Simulator in supported:
        if name==Simulator.name:
            return Simulator
    raise NotImplementedError(f'Simulator {name} not found')
