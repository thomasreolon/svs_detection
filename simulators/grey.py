from .forensor_sim import StaticSVS

"""
instead of applying SVS algorithm returns input image
- useful for comparing how much applying SVS loses information wrt original image
"""

class GreyscaleSVS(StaticSVS):
    name = 'grey'
    def __call__(self, frame):
        return frame


