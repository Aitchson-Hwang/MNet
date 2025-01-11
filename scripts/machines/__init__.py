from .pytorch_ssim import SSIM
from .BasicMachine import BasicMachine
from .VX import VX
from .S2AM import S2AM
from .SLBR import SLBR
from .SUNET import SUNet
from .TestAllMethods import AllMethods
from .DENet import DENet
def basic(**kwargs):
	return BasicMachine(**kwargs)

def s2am(**kwargs):
    return S2AM(**kwargs)

def vx(**kwargs):
    return VX(**kwargs)

def sunet(**kwargs):
    return SUNet(**kwargs)

def testallmethods(**kwargs):
    return AllMethods(**kwargs)

def slbr(**kwargs):
    return SLBR(**kwargs)

def denet(**kwargs):
    return DENet(**kwargs)


