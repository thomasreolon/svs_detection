from .forensor_sim import StaticSVS
from .grey import GreyscaleSVS
from .rlearn import RLearnSVS
from.mhi import MHISVS 
from.rlearnmhi import MHIRLearnSVS 
from.mhigrey import MHIGreySVS 

supported = [
    StaticSVS,      # static (baseline)
    GreyscaleSVS,   # grey (baseline)
    RLearnSVS,      # policy
    MHISVS,         # motion history image
    MHIGreySVS,     # mhi & greyscale
    MHIRLearnSVS,   # policy with motion history image
]

def get_simulator(name, svs_close, svs_open, svs_hot, svs_ker, policy):
    Class = None
    for Simulator in supported:
        if name==Simulator.name:
            Class = Simulator
            break
    
    if Class is None:
        raise NotImplementedError(f'Simulator {name} not found')
    
    if 'policy' not in name:
        return Class(svs_close, svs_open, svs_hot, svs_ker)
    else:
        return Class(svs_close, svs_open, svs_hot, svs_ker, policy)
