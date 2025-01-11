from scripts.models.rasc import *
from scripts.models.unet import UnetGenerator,MinimalUnetV2
from scripts.models.vmu import UnetVM
from scripts.models.sa_resunet import UnetVMS2AMv4
from scripts.models.slbr import SLBR
from scripts.mynets.MNet import MNet
from scripts.models.denet import DENet
from scripts.mynets.MNetold import MNetold
# our method
def vqgan(**kwargs):
    return VQVAE()
def mnet(**kwargs):
    return MNet(wf=54, scale=42, vscale=42, args=kwargs['args'])
def mnetold(**kwargs):
    return MNetold(wf=54, scale=42, vscale=42, args=kwargs['args'])
# splitnet is vvv4n
def vvv4n(**kwargs):
    return UnetVMS2AMv4(shared_depth=2, blocks=3, long_skip=True, use_vm_decoder=True,s2am='vms2am')
def slbr(**kwargs):
    return SLBR(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)
def denet(**kwargs):
    return DENet(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)
# BVMR
def vm3(**kwargs):
    return UnetVM(shared_depth=2, blocks=3, use_vm_decoder=True)
# Blind version of S2AM
def urasc(**kwargs):
    model = UnetGenerator(3,3,is_attention_layer=True,attention_model=URASC,basicblock=MinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model
# Improving the Harmony of the Composite Image by Spatial-Separated Attention Module
# Xiaodong Cun and Chi-Man Pun
# University of Macau
# Trans. on Image Processing, vol. 29, pp. 4759-4771, 2020.
def rascv2(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=RASC,basicblock=MinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model
# just original unet
def unet(**kwargs):
    model = UnetGenerator(3,3)
    model.apply(weights_init_kaiming)
    return model


