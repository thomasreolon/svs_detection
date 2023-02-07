from .forensor_sim import StaticSVS
from .grey import GreyscaleSVS

supported = [
    StaticSVS,      # static
    GreyscaleSVS    # grey
]

def get_simulator(name):
    for Simulator in supported:
        if name==Simulator.name:
            return Simulator
    raise NotImplementedError(f'Simulator {name} not found')
