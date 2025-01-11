
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.blocks_slbr import UpConv, DownConv, MBEBlock, SMRBlock, CFFBlock, ResDownNew, ResUpNew, ECABlock
import scipy.stats as st
import itertools
import cv2

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)

class CoarseEncoder(nn.Module):
    def __init__(self, in_channels=3, depth=3, blocks=1, start_filters=32, residual=True, norm=nn.BatchNorm2d, act=F.relu):
        super(CoarseEncoder, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters*(2**i)
            # pooling = True if i < depth-1 else False
            pooling = True
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, residual=residual, norm=norm, act=act)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs

class SharedBottleNeck(nn.Module):
    def __init__(self, in_channels=512, depth=5, shared_depth=2, start_filters=32, blocks=1, residual=True,
                 concat=True,  norm=nn.BatchNorm2d, act=F.relu, dilations=[1,2,5]):
        super(SharedBottleNeck, self).__init__()
        self.down_convs = []
        self.up_convs = []
        self.down_mask_atts = []
        self.up_mask_atts = []

        dilations = [1,2,5]
        start_depth = depth - shared_depth
        max_filters = 512
        for i in range(start_depth, depth): # depth = 5 [0,1,2,3]
            ins = in_channels if i == start_depth else outs
            outs = min(ins * 2, max_filters)
            # Encoder convs
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, residual=residual, norm=norm, act=act, dilations=dilations)
            self.down_convs.append(down_conv)

            # Decoder convs
            if i < depth - 1:
                up_conv = UpConv(min(outs*2, max_filters), outs, blocks, residual=residual, concat=concat, norm=norm,act=F.relu, dilations=dilations)
                self.up_convs.append(up_conv)
                self.up_mask_atts.append(ECABlock(outs))
       
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        # task-specific channel attention blocks
        self.up_mask_atts = nn.ModuleList(self.up_mask_atts)

        reset_params(self)

    def forward(self, input):
        # Encoder convs
        mask_encoder_outs = []
        x = input
        for i, d_conv in enumerate(self.down_convs):
            # d_conv, attn = nets
            x, before_pool = d_conv(x)
            mask_encoder_outs.append(before_pool)
        x_mask = x

        # Decoder convs

        x = x_mask       
        for i, nets in enumerate(zip(self.up_convs, self.up_mask_atts)):
            up_conv, attn = nets
            before_pool = None
            if mask_encoder_outs is not None:
                before_pool = mask_encoder_outs[-(i+2)]
            x = up_conv(x, before_pool, se = attn)
        x_mask = x

        return x_mask

class CoarseDecoder(nn.Module):
    def __init__(self, args, in_channels=512, out_channels=3, norm='bn',act=F.relu, depth=5, blocks=1, residual=True,
                 concat=True, use_att=False):
        super(CoarseDecoder, self).__init__()
        self.up_convs_mask = []
        self.up_convs_bg = []

        # apply channel attention to skip connection for different decoders
        self.atts_mask = []
        self.use_att = use_att
        outs = in_channels
        for i in range(depth): 
            ins = outs
            outs = ins // 2
            up_conv = MBEBlock(args.bg_mode, ins, outs, blocks=blocks, residual=residual, concat=concat, norm='in', act=act)
            self.up_convs_bg.append(up_conv)
            # mask prediction branch
            up_conv = SMRBlock(args, ins, outs, blocks=blocks, residual=residual, concat=concat, norm=norm, act=act)
            self.up_convs_mask.append(up_conv)
            if self.use_att:
                self.atts_mask.append(ECABlock(outs))
        # final conv
        self.up_convs_bg = nn.ModuleList(self.up_convs_bg)
        self.conv_final_bg = nn.Conv2d(outs, out_channels, 1,1,0)
        self.up_convs_mask = nn.ModuleList(self.up_convs_mask)
        self.atts_mask = nn.ModuleList(self.atts_mask)
        
        reset_params(self)

    def forward(self, mask, encoder_outs=None):
        mask_x = mask
        mask_outs = []
        for i, up_convs in enumerate(zip(self.up_convs_bg, self.up_convs_mask)):
            up_bg, up_mask = up_convs
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+1)]

            if self.use_att:
                mask_before_pool = self.atts_mask[i](before_pool)
            smr_outs = up_mask(mask_x, mask_before_pool)
            mask_x= smr_outs['feats'][0]
            primary_map, self_calibrated_map = smr_outs['attn_maps']
            mask_outs.append(primary_map)
            mask_outs.append(self_calibrated_map)



        if self.conv_final_bg is not None:
            mask_x = mask_outs[-1]
        return [mask_x] + mask_outs


 



class DiceSLBR(nn.Module):

    def __init__(self, args, in_channels=3, depth=5, shared_depth=2, blocks=1,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True,
                 concat=True, long_skip=False):
        super(DiceSLBR, self).__init__()
        self.shared = shared_depth = 2
        self.optimizer_encoder,  self.optimizer_image, self.optimizer_wm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        self.args = args
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks, blocks)

        # coarse stage
        self.encoder = CoarseEncoder(in_channels=in_channels, depth= depth - shared_depth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, norm='bn',act=F.relu)
        self.shared_decoder = SharedBottleNeck(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                               depth=depth, shared_depth=shared_depth, blocks=blocks[4], residual=residual,
                                                concat=concat, norm='in')
        
        self.coarse_decoder = CoarseDecoder(args, in_channels=start_filters * 2 ** (depth - shared_depth),
                                        out_channels=out_channels_image, depth=depth - shared_depth,
                                        blocks=blocks[1], residual=residual, 
                                        concat=concat, norm='bn', use_att=True,
                                        )

        self.long_skip = long_skip
        self.refinement = None

    def set_optimizers(self):
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=self.args.lr)
        self.optimizer_image = torch.optim.Adam(self.coarse_decoder.parameters(), lr=self.args.lr)
        
        if self.refinement is not None:
            self.optimizer_refine = torch.optim.Adam(self.refinement.parameters(), lr=self.args.lr)
        
        if self.shared != 0:
            self.optimizer_shared = torch.optim.Adam(self.shared_decoder.parameters(), lr=self.args.lr)

    def zero_grad_all(self):
        self.optimizer_encoder.zero_grad()
        self.optimizer_image.zero_grad()
        
        if self.shared != 0:
            self.optimizer_shared.zero_grad()
        if self.refinement is not None:
            self.optimizer_refine.zero_grad()

    def step_all(self):
        self.optimizer_encoder.step()
        if self.shared != 0:
               self.optimizer_shared.step()
        self.optimizer_image.step()
        if self.refinement is not None:
            self.optimizer_refine.step()

    def multi_gpu(self):
        self.encoder = nn.DataParallel(self.encoder, device_ids=range(torch.cuda.device_count()))
        self.shared_decoder = nn.DataParallel(self.shared_decoder, device_ids=range(torch.cuda.device_count()))
        self.coarse_decoder = nn.DataParallel(self.coarse_decoder, device_ids=range(torch.cuda.device_count()))
        if self.refinement is not None:
            self.refinement = nn.DataParallel(self.refinement, device_ids=range(torch.cuda.device_count()))
        return

    def forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        unshared_before_pool = before_pool #[: - self.shared]

        mask = self.shared_decoder(image_code)
        mask = self.coarse_decoder(mask, unshared_before_pool)

        return mask


