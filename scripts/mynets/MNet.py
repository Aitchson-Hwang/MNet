"""
## Multi-Scale-Stage Network
## Code is based on Multi-Stage Progressive Image Restoration(MPRNet) and MSSNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

##########################################################################
## Residual Block
class ResBlock(nn.Module):
    def __init__(self, wf, kernel_size, reduction, bias, act):
        super(ResBlock, self).__init__()
        modules_body = []
        modules_body.append(conv(wf, wf, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(wf, wf, kernel_size, bias=bias))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, wf, scale, vscale, kernel_size, reduction, act, bias, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [ResBlock(wf,              kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [ResBlock(wf+scale,        kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [ResBlock(wf+scale+vscale, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(wf, scale)
        self.down23  = DownSample(wf+scale, vscale)

        if csff:
            self.csff_enc1 = nn.Conv2d(wf,              wf,              kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(wf+scale,        wf+scale,        kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(wf+scale+vscale, wf+scale+vscale, kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(wf,              wf,              kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(wf+scale,        wf+scale,        kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(wf+scale+vscale, wf+scale+vscale, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, wf, scale, vscale, kernel_size, reduction, act, bias):
        super(Decoder, self).__init__()

        self.decoder_level1 = [ResBlock(wf,              kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [ResBlock(wf+scale,        kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [ResBlock(wf+scale+vscale, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = ResBlock(wf,       kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = ResBlock(wf+scale, kernel_size, reduction, bias=bias, act=act)

        self.up32  = SkipUpSample(wf+scale, vscale)
        self.up21  = SkipUpSample(wf, scale)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class EDUp(nn.Module):
    def __init__(self):
        super(EDUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, elist, dlist):

        up_elist = []
        for feat in elist:
            up_elist.append(self.up(feat))

        up_dlist = []
        for feat in dlist:
            up_dlist.append(self.up(feat))

        return up_elist, up_dlist

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class ScaleUpSample(nn.Module):
    def __init__(self, in_channels):
        super(ScaleUpSample,self).__init__()
        self.scaleUp = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                     nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))
    def forward(self,feat):
        return self.scaleUp(feat)

#https://github.com/fangwei123456/PixelUnshuffle-pytorch
class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * (k*w) * (k*h)
        kdownscale_factor: k
        batchSize * c * (k*w) * (k*h) -> batchSize * (k*k*c) * w * h
        '''
        c = input.shape[1]

        kernel = torch.zeros(size=[self.downscale_factor * self.downscale_factor * c,
                                   1, self.downscale_factor, self.downscale_factor],
                             device=input.device)
        for y in range(self.downscale_factor):
            for x in range(self.downscale_factor):
                kernel[x + y * self.downscale_factor::self.downscale_factor*self.downscale_factor, 0, y, x] = 1
        return F.conv2d(input, kernel, stride=self.downscale_factor, groups=c)

class Tail_shuffle(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,bias):
        super(Tail_shuffle,self).__init__()
        self.tail = conv(in_channels, out_channels, kernel_size, bias=bias)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, feat):
        return self.pixelshuffle(self.tail(feat))

##########################################################################

class MNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, wf=54, scale=42, vscale=42, kernel_size=3, reduction=4, bias=False, args = None):
        super(MNet, self).__init__()
        self.k1 = args.k1 # layer 1
        self.k2 = args.k2 # layer 2
        self.k3 = args.k3 # layer 3
        self.srm = args.srm
        self.inf = args.inf
        self.crf = args.crf
        self.sharing = args.sharing
        act=nn.PReLU()
        self.pixel_unshuffle = PixelUnshuffle(2)
        self.ED_up = EDUp()
        # the following are the UNets setting in various layers
        # layer 3
        if self.k3 >= 1:
            self.shallow_feat1 = nn.Sequential(conv(12, wf, kernel_size, bias=bias), ResBlock(wf,kernel_size, reduction, bias=bias, act=act))
            self.E1_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=False)
            self.D1_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            # self.mE1_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=False)
            # self.mD1_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            if self.sharing == 0: # 一部分模型需要注释，而保留上面的部分
                self.mE1_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=False)
                self.mD1_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            self.tail1_1 = Tail_shuffle(wf, 12, kernel_size, bias=bias)

            if self.k3 >= 2:
                self.E1_2 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                self.D1_2 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            if self.k3 >= 3:
                self.E1_3 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                self.D1_3 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        ##########################################################################
        # layer 2
        if self.k2 >= 1:
            if self.srm == 1:
                self.shallow_feat2 = nn.Sequential(conv(12, wf, kernel_size, bias=bias), ResBlock(wf,kernel_size, reduction, bias=bias, act=act))
            self.up_scale1_feat = ScaleUpSample(wf)
            self.fusion12 = conv(wf*2, wf, kernel_size, bias=bias)
            self.mup_scale1_feat = ScaleUpSample(wf)
            self.mfusion12 = conv(wf*2, wf, kernel_size, bias=bias)
            self.E2_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.D2_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            # self.mE2_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            # self.mD2_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            if self.sharing == 0:
                self.mE2_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                self.mD2_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

            if self.k2 >= 2:
                self.E2_2 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                self.D2_2 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
                # self.mE2_2 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                # self.mD2_2 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
                if self.sharing == 0:  # 部分模型可能要注释掉
                    self.mE2_2 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                    self.mD2_2 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            if self.k2 >= 3:
                self.E2_3 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                self.D2_3 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
                self.mE2_3 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                self.mD2_3 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            if self.k2 >= 4:
                self.E2_4 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                self.D2_4 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
                self.mE2_4 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                self.mD2_4 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        ################################################################################
        # layer 1
        if self.srm == 1:
            self.shallow_feat3 = nn.Sequential(conv(3, wf, kernel_size, bias=bias), ResBlock(wf,kernel_size, reduction, bias=bias, act=act))
        self.up_scale2_feat = ScaleUpSample(wf)
        self.fusion23 = conv(wf*2, wf, kernel_size, bias=bias)

        self.E3_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D3_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        if self.k3 != 0 or self.k2 != 0 or self.sharing != 1:
            self.mE3_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.mD3_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        if self.k1 >= 2:
            self.E3_2 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.D3_2 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            if self.k3 != 0 or self.k2 != 0 or self.sharing != 1:
                self.mE3_2 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                self.mD3_2 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        if self.k1 >= 3:
            self.E3_3 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.D3_3 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            if self.k3 != 0 or self.k2 != 0 or self.sharing != 1:
                self.mE3_3 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
                self.mD3_3 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        if self.k1 >= 4:
            self.E3_4 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.D3_4 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            self.mE3_4 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.mD3_4 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        if self.k1 >= 5:
            self.E3_5 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.D3_5 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            self.mE3_5 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.mD3_5 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        if self.k1 >= 6:
            self.E3_6 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.D3_6 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            self.mE3_6 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.mD3_6 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        if self.k1 >= 7:
            self.E3_7 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.D3_7 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            self.mE3_7 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.mD3_7 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        if self.k1 >= 8:
            self.E3_8 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.D3_8 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
            self.mE3_8 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
            self.mD3_8 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        self.tail3 = conv(wf, 3, kernel_size, bias=bias)


        self.mup_scale2_feat = ScaleUpSample(wf)
        self.mfusion23 = conv(wf*2, wf, kernel_size, bias=bias)
        self.mtail3 = conv(wf, 1, kernel_size, bias=bias)
        self.optimizer = None
    def set_optimizers(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    def step_all(self):
        self.optimizer.step()
    def zero_grad_all(self):
        self.optimizer.zero_grad()
    def forward(self, s3_blur):

        interpolation = nn.Upsample(scale_factor=0.5, mode = 'bilinear',align_corners=True)
        s2_blur = interpolation(s3_blur)
        e_list, d_list = None, None # cross-layer feature fusion
        me_list, md_list = None, None # cross-layer feature fusion
        ##-------------------------------------------
        ##-------------- layer 3---------------------
        ##-------------------------------------------
        if self.k3 != 0:
            s1_blur_ps = self.pixel_unshuffle(s2_blur)
            shfeat1 = self.shallow_feat1(s1_blur_ps)       

            e1_1f = self.E1_1(shfeat1)
            d1_1f = self.D1_1(e1_1f)
            if self.sharing == 0:
                me1_1f = self.mE1_1(shfeat1)
                md1_1f = self.mD1_1(me1_1f)
            else:
                me1_1f = self.E1_1(shfeat1)
                md1_1f = self.D1_1(me1_1f)

            if self.k3 >= 2:
                if self.inf == 1:
                    e1_2f = self.E1_2(d1_1f[0],e1_1f,d1_1f)
                    me1_2f = self.E1_2(md1_1f[0], me1_1f, md1_1f)
                else:
                    e1_2f = self.E1_2(d1_1f[0],None,None)
                    me1_2f = self.E1_2(md1_1f[0], None, None)

                d1_2f = self.D1_2(e1_2f)
                md1_2f = self.D1_2(me1_2f)

            if self.k3 >= 3:
                if self.inf == 1:
                    e1_3f = self.E1_3(d1_2f[0],e1_2f,d1_2f)
                    me1_3f = self.E1_3(md1_2f[0], me1_2f, md1_2f)
                else:
                    e1_3f = self.E1_3(d1_2f[0],None,None)
                    me1_3f = self.E1_3(md1_2f[0], None, None)

                d1_3f = self.D1_3(e1_3f)
                md1_3f = self.D1_3(me1_3f)

        ##-------------------------------------------
        ##-------------- layer 2---------------------
        ##-------------------------------------------
        if self.k2 != 0:
            s2_blur_ps = self.pixel_unshuffle(s3_blur)
            shfeat2 = self.shallow_feat2(s2_blur_ps)

            if self.k3 == 1:
                s1_sol_feat = self.up_scale1_feat(d1_1f[0])
                fusion12 = self.fusion12(torch.cat([shfeat2,s1_sol_feat],1))
                ms1_sol_feat = self.up_scale1_feat(md1_1f[0])
                mfusion12 = self.fusion12(torch.cat([shfeat2,ms1_sol_feat],1))
                if self.crf == 1:
                    e_list, d_list = self.ED_up(e1_1f, d1_1f)
                    me_list, md_list = self.ED_up(me1_1f, md1_1f)
            elif self.k3 == 2:
                s1_sol_feat = self.up_scale1_feat(d1_2f[0])
                fusion12 = self.fusion12(torch.cat([shfeat2,s1_sol_feat],1))
                ms1_sol_feat = self.up_scale1_feat(md1_2f[0])
                mfusion12 = self.fusion12(torch.cat([shfeat2,ms1_sol_feat],1))
                if self.crf == 1:
                    e_list, d_list = self.ED_up(e1_2f, d1_2f)
                    me_list, md_list = self.ED_up(me1_2f, md1_2f)

            elif self.k3 == 3:
                s1_sol_feat = self.up_scale1_feat(d1_3f[0])
                fusion12 = self.fusion12(torch.cat([shfeat2,s1_sol_feat],1))
                ms1_sol_feat = self.up_scale1_feat(md1_3f[0])
                mfusion12 = self.fusion12(torch.cat([shfeat2,ms1_sol_feat],1))
                if self.crf == 1:
                    e_list, d_list = self.ED_up(e1_3f, d1_3f)
                    me_list, md_list = self.ED_up(me1_3f, md1_3f)
            elif self.k3 == 0:
                fusion12 = shfeat2
                mfusion12 = shfeat2

            e2_1f = self.E2_1(fusion12,e_list,d_list)
            d2_1f = self.D2_1(e2_1f)
            if self.sharing == 1:
                me2_1f = self.E2_1(mfusion12, me_list, md_list)
                md2_1f = self.D2_1(me2_1f)
            else:
                me2_1f = self.E2_1(mfusion12, me_list, md_list)
                md2_1f = self.mD2_1(me2_1f)
            if self.k2 >= 2:
                if self.inf == 1:
                    e2_2f = self.E2_2(d2_1f[0],e2_1f,d2_1f)
                    if self.sharing == 1:
                        me2_2f = self.E2_2(md2_1f[0], me2_1f, md2_1f)
                    else:
                        me2_2f = self.mE2_2(md2_1f[0], me2_1f, md2_1f)
                else:
                    e2_2f = self.E2_2(d2_1f[0], None, None)
                    if self.sharing == 1:
                        me2_2f = self.E2_2(md2_1f[0], None, None)
                    else:
                        me2_2f = self.mE2_2(md2_1f[0], None, None)
                d2_2f = self.D2_2(e2_2f)
                if self.sharing == 1:
                    md2_2f = self.D2_2(me2_2f)
                else:
                    md2_2f = self.mD2_2(me2_2f)
            if self.k2 >= 3:
                if self.inf == 1:
                    e2_3f = self.E2_3(d2_2f[0], e2_2f, d2_2f)
                    me2_3f = self.E2_3(d2_2f[0], me2_2f, md2_2f)
                else:
                    e2_3f = self.E2_3(d2_2f[0],None,None)
                    me2_3f = self.E2_3(d2_2f[0], None, None)
                d2_3f = self.D2_3(e2_3f)
                md2_3f = self.D2_3(me2_3f)
            if self.k2 >= 4:
                if self.inf == 1:
                    e2_4f = self.E2_4(d2_3f[0], e2_3f, d2_3f)
                    me2_4f = self.E2_4(d2_3f[0], me2_3f, md2_3f)
                else:
                    e2_4f = self.E2_4(d2_3f[0],None,None)
                    me2_4f = self.E2_4(d2_3f[0], None, None)
                d2_4f = self.D2_4(e2_4f)
                md2_4f = self.D2_4(me2_4f)
        ##-------------------------------------------
        ##-------------- layer 1---------------------
        ##-------------------------------------------
        shfeat3 = self.shallow_feat3(s3_blur)

        if self.k2 == 0:
            fusion23 = shfeat3
            e_list, d_list = None, None
            mfusion23 = shfeat3
            me_list, md_list = None, None
        elif self.k2 == 1:
            s2_sol_feat = self.up_scale2_feat(d2_1f[0])
            fusion23 = self.fusion23(torch.cat([shfeat3, s2_sol_feat], 1))
            if self.crf == 1:
                e_list, d_list = self.ED_up(e2_1f, d2_1f)
            ms2_sol_feat = self.up_scale2_feat(md2_1f[0])
            mfusion23 = self.fusion23(torch.cat([shfeat3, ms2_sol_feat], 1))
            if self.crf == 1:
                me_list, md_list = self.ED_up(me2_1f, md2_1f)
        elif self.k2 == 2:
            s2_sol_feat = self.up_scale2_feat(d2_2f[0])
            fusion23 = self.fusion23(torch.cat([shfeat3, s2_sol_feat], 1))
            if self.crf == 1:
                e_list, d_list = self.ED_up(e2_2f, d2_2f)
            ms2_sol_feat = self.up_scale2_feat(md2_2f[0])
            mfusion23 = self.fusion23(torch.cat([shfeat3, ms2_sol_feat], 1))
            if self.crf == 1:
                me_list, md_list = self.ED_up(me2_2f, md2_2f)
        elif self.k2 == 3:
            s2_sol_feat = self.up_scale2_feat(d2_3f[0])
            fusion23 = self.fusion23(torch.cat([shfeat3, s2_sol_feat], 1))
            if self.crf == 1:
                e_list, d_list = self.ED_up(e2_3f, d2_3f)
            ms2_sol_feat = self.up_scale2_feat(md2_3f[0])
            mfusion23 = self.fusion23(torch.cat([shfeat3, ms2_sol_feat], 1))
            if self.crf == 1:
                me_list, md_list = self.ED_up(me2_3f, md2_3f)
        elif self.k2 == 4:
            s2_sol_feat = self.up_scale2_feat(d2_4f[0])
            fusion23 = self.fusion23(torch.cat([shfeat3, s2_sol_feat], 1))
            if self.crf == 1:
                e_list, d_list = self.ED_up(e2_4f, d2_4f)
            ms2_sol_feat = self.up_scale2_feat(md2_4f[0])
            mfusion23 = self.fusion23(torch.cat([shfeat3, ms2_sol_feat], 1))
            if self.crf == 1:
                me_list, md_list = self.ED_up(me2_4f, md2_4f)


        e3_1f = self.E3_1(fusion23,e_list,d_list)
        d3_1f = self.D3_1(e3_1f)
        if self.k3 != 0 or self.k2 != 0 or self.sharing != 1:
            me3_1f = self.mE3_1(mfusion23, me_list, md_list)
            md3_1f = self.mD3_1(me3_1f)
        else:
            me3_1f = self.E3_1(fusion23,e_list,d_list)
            md3_1f = self.D3_1(me3_1f)

        if self.inf == 1:
            if self.k1 >= 2:
                e3_2f = self.E3_2(d3_1f[0],e3_1f,d3_1f)
                d3_2f = self.D3_2(e3_2f)
                if self.k3 != 0 or self.k2 != 0 or self.sharing != 1:
                    me3_2f = self.mE3_2(md3_1f[0], me3_1f, md3_1f)
                    md3_2f = self.mD3_2(me3_2f)
                else:
                    me3_2f = self.E3_2(md3_1f[0], me3_1f, md3_1f)
                    md3_2f = self.D3_2(me3_2f)
            if self.k1 >= 3:
                e3_3f = self.E3_3(d3_2f[0],e3_2f,d3_2f)
                d3_3f = self.D3_3(e3_3f)
                if self.k3 != 0 or self.k2 != 0 or self.sharing != 1:
                    me3_3f = self.mE3_3(md3_2f[0], me3_2f, md3_2f)
                    md3_3f = self.mD3_3(me3_3f)
                else:
                    me3_3f = self.E3_3(md3_2f[0], me3_2f, md3_2f)
                    md3_3f = self.D3_3(me3_3f)
            if self.k1 >= 4:
                e3_4f = self.E3_4(d3_3f[0],e3_3f,d3_3f)
                d3_4f = self.D3_4(e3_4f)
                me3_4f = self.mE3_4(md3_3f[0], me3_3f, md3_3f)
                md3_4f = self.mD3_4(me3_4f)
            if self.k1 >= 5:
                e3_5f = self.E3_5(d3_4f[0],e3_4f,d3_4f)
                d3_5f = self.D3_5(e3_5f)
                me3_5f = self.mE3_5(md3_4f[0], me3_4f, md3_4f)
                md3_5f = self.mD3_5(me3_5f)
            if self.k1 >= 6:
                e3_6f = self.E3_6(d3_5f[0],e3_5f,d3_5f)
                d3_6f = self.D3_6(e3_6f)
                me3_6f = self.mE3_6(md3_5f[0], me3_5f, md3_5f)
                md3_6f = self.mD3_6(me3_6f)
            if self.k1 >= 7:
                e3_7f = self.E3_7(d3_6f[0],e3_6f,d3_6f)
                d3_7f = self.D3_7(e3_7f)
                me3_7f = self.mE3_7(md3_6f[0], me3_6f, md3_6f)
                md3_7f = self.mD3_7(me3_7f)
            if self.k1 >= 8:
                e3_8f = self.E3_8(d3_7f[0],e3_7f,d3_7f)
                d3_8f = self.D3_8(e3_8f)
                me3_8f = self.mE3_8(md3_7f[0], me3_7f, md3_7f)
                md3_8f = self.mD3_8(me3_8f)
        else:
            if self.k1 >= 2:
                e3_2f = self.E3_2(d3_1f[0],None,None)
                d3_2f = self.D3_2(e3_2f)
                if self.k3 != 0 or self.k2 != 0 or self.sharing != 1:
                    me3_2f = self.mE3_2(md3_1f[0], None, None)
                    md3_2f = self.mD3_2(me3_2f)
                else:
                    me3_2f = self.E3_2(md3_1f[0], None, None)
                    md3_2f = self.D3_2(me3_2f)
            if self.k1 >= 3:
                e3_3f = self.E3_3(d3_2f[0],None,None)
                d3_3f = self.D3_3(e3_3f)
                if self.k3 != 0 or self.k2 != 0 or self.sharing != 1:
                    me3_3f = self.mE3_3(md3_2f[0], None, None)
                    md3_3f = self.mD3_3(me3_3f)
                else:
                    me3_3f = self.E3_3(md3_2f[0], None, None)
                    md3_3f = self.D3_3(me3_3f)
            if self.k1 >= 4:
                e3_4f = self.E3_4(d3_3f[0],None,None)
                d3_4f = self.D3_4(e3_4f)
                me3_4f = self.mE3_4(md3_3f[0], None, None)
                md3_4f = self.mD3_4(me3_4f)
            if self.k1 >= 5:
                e3_5f = self.E3_5(d3_4f[0],None,None)
                d3_5f = self.D3_5(e3_5f)
                me3_5f = self.mE3_5(md3_4f[0], None, None)
                md3_5f = self.mD3_5(me3_5f)
            if self.k1 >= 6:
                e3_6f = self.E3_6(d3_5f[0],None,None)
                d3_6f = self.D3_6(e3_6f)
                me3_6f = self.mE3_6(md3_5f[0], None, None)
                md3_6f = self.mD3_6(me3_6f)
            if self.k1 >= 7:
                e3_7f = self.E3_7(d3_6f[0],None,None)
                d3_7f = self.D3_7(e3_7f)
                me3_7f = self.mE3_7(md3_6f[0], None, None)
                md3_7f = self.mD3_7(me3_7f)
            if self.k1 >= 8:
                e3_8f = self.E3_8(d3_7f[0],None,None)
                d3_8f = self.D3_8(e3_8f)
                me3_8f = self.mE3_8(md3_7f[0], None, None)
                md3_8f = self.mD3_8(me3_8f)

        if self.k1 == 1:
            res_bg = self.tail3(d3_1f[0]) + s3_blur
        elif self.k1 == 2:
            res_bg = self.tail3(d3_2f[0]) + s3_blur
        elif self.k1 == 3:
            res_bg = self.tail3(d3_3f[0]) + s3_blur
        elif self.k1 == 4:
            res_bg = self.tail3(d3_4f[0]) + s3_blur
        elif self.k1 == 5:
            res_bg = self.tail3(d3_5f[0]) + s3_blur
        elif self.k1 == 6:
            res_bg = self.tail3(d3_6f[0]) + s3_blur
        elif self.k1 == 7:
            res_bg = self.tail3(d3_7f[0]) + s3_blur
        elif self.k1 == 8:
            res_bg = self.tail3(d3_8f[0]) + s3_blur

        if self.k1 == 1:
            mask = torch.sigmoid(self.mtail3(md3_1f[0]))
        elif self.k1 == 2:
            mask = torch.sigmoid(self.mtail3(md3_2f[0]))
        elif self.k1 == 3:
            mask = torch.sigmoid(self.mtail3(md3_3f[0]))
        elif self.k1 == 4:
            mask = torch.sigmoid(self.mtail3(md3_4f[0]))
        elif self.k1 == 5:
            mask = torch.sigmoid(self.mtail3(md3_5f[0]))
        elif self.k1 == 6:
            mask = torch.sigmoid(self.mtail3(md3_6f[0]))
        elif self.k1 == 7:
            mask = torch.sigmoid(self.mtail3(md3_7f[0]))
        elif self.k1 == 8:
            mask = torch.sigmoid(self.mtail3(md3_8f[0]))
        return res_bg, mask

