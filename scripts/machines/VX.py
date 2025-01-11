import csv

import torch
import torch.nn as nn
from progress.bar import Bar
from torch import optim
from tqdm import tqdm
# import pytorch_ssim
from scripts.machines import pytorch_ssim
import json
import sys,time,os
import torchvision
from math import log10
import numpy as np
from .BasicMachine import BasicMachine
from scripts.utils.evaluation import accuracy, AverageMeter, final_preds
from scripts.utils.misc import resize_to_match
from torch.autograd import Variable
import torch.nn.functional as F
from scripts.utils.parallel import DataParallelModel, DataParallelCriterion
from scripts.utils.losses import VGGLoss, l1_relative,is_dic
from scripts.utils.imutils import im_to_numpy
import skimage.io
from skimage.measure import compare_psnr,compare_ssim
import lpips
from datetime import datetime

class Losses(nn.Module):
    def __init__(self, argx, device, norm_func=None, denorm_func=None):
        super(Losses, self).__init__()
        self.args = argx

        if self.args.loss_type == 'l1bl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.L1Loss(), nn.BCELoss(), nn.MSELoss()
        elif self.args.loss_type == 'l2xbl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.MSELoss(), nn.BCEWithLogitsLoss(), nn.MSELoss()
        elif self.args.loss_type == 'relative' or self.args.loss_type == 'hybrid':
            self.outputLoss, self.attLoss, self.wrloss = l1_relative, nn.BCELoss(), l1_relative
        else: # l2bl2
            self.outputLoss, self.attLoss, self.wrloss = nn.MSELoss(), nn.BCELoss(), nn.MSELoss()

        self.default = nn.L1Loss()

        if self.args.style_loss > 0:
            self.vggloss = VGGLoss(self.args.sltype).to(device)
        
        if self.args.ssim_loss > 0:
            self.ssimloss =  pytorch_ssim.SSIM().to(device)
        
        self.norm = norm_func
        self.denorm = denorm_func


    def forward(self,pred_ims,target,pred_ms,mask,pred_wms,wm):
        pixel_loss,att_loss,wm_loss,vgg_loss,ssim_loss = [0]*5
        pred_ims = pred_ims if is_dic(pred_ims) else [pred_ims]

        # try the loss in the masked region
        if self.args.masked and 'hybrid' in self.args.loss_type: # masked loss
            pixel_loss += sum([self.outputLoss(pred_im, target, mask) for pred_im in pred_ims])
            pixel_loss += sum([self.default(pred_im*pred_ms,target*mask) for pred_im in pred_ims])
            recov_imgs = [ self.denorm(pred_im*mask + (1-mask)*self.norm(target)) for pred_im in pred_ims ]
            wm_loss += self.wrloss(pred_wms, wm, mask)
            wm_loss += self.default(pred_wms*pred_ms, wm*mask)

        elif self.args.masked and 'relative' in self.args.loss_type: # masked loss
            pixel_loss += sum([self.outputLoss(pred_im, target, mask) for pred_im in pred_ims])
            recov_imgs = [ self.denorm(pred_im*mask + (1-mask)*self.norm(target)) for pred_im in pred_ims ]
            wm_loss = self.wrloss(pred_wms, wm, mask)
        elif self.args.masked:
            pixel_loss += sum([self.outputLoss(pred_im*mask, target*mask) for pred_im in pred_ims])
            recov_imgs = [ self.denorm(pred_im*pred_ms + (1-pred_ms)*self.norm(target)) for pred_im in pred_ims ]
            wm_loss = self.wrloss(pred_wms*mask, wm*mask)
        else:
            pixel_loss += sum([self.outputLoss(pred_im*pred_ms, target*mask) for pred_im in pred_ims])
            recov_imgs = [ self.denorm(pred_im*pred_ms + (1-pred_ms)*self.norm(target)) for pred_im in pred_ims ]
            wm_loss = self.wrloss(pred_wms*pred_ms,wm*mask)

        pixel_loss += sum([self.default(im,target) for im in recov_imgs])

        if self.args.style_loss > 0:
            vgg_loss = sum([self.vggloss(im,target,mask) for im in recov_imgs])

        if self.args.ssim_loss > 0:
            ssim_loss = sum([ 1 - self.ssimloss(im,target) for im in recov_imgs])

        att_loss =  self.attLoss(pred_ms, mask)

        return pixel_loss,att_loss,wm_loss,vgg_loss,ssim_loss


class VX(BasicMachine):
    def __init__(self,**kwargs):
        BasicMachine.__init__(self,**kwargs)
        self.loss = Losses(self.args, self.device, self.norm, self.denorm)
        self.model.set_optimizers()  # 单GPU
        # self.model.module.set_optimizers() # 多GPU
        self.optimizer = None

       
    def train(self,epoch):

        self.current_epoch = epoch

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        lossMask = AverageMeter()
        lossWM = AverageMeter()
        lossMX = AverageMeter()
        lossvgg = AverageMeter()
        lossssim = AverageMeter()
        # switch to train mode
        self.model.train()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))

        for i, batches in enumerate(self.train_loader):

            current_index = len(self.train_loader) * epoch + i

            inputs = batches['image'].to(self.device)
            target = batches['target'].to(self.device)
            mask = batches['mask'].to(self.device)
            wm = batches['wm'].to(self.device)
            outputs = self.model(self.norm(inputs))
            
            self.model.zero_grad_all()
            # outputs[0]是估计的背景图像，outputs[1]是mask,outputs[2]是水印图像

            l2_loss,att_loss,wm_loss,style_loss,ssim_loss = self.loss(outputs[0],self.norm(target),outputs[1],mask,outputs[2],self.norm(wm))
            total_loss = 2*l2_loss + self.args.att_loss * att_loss + wm_loss + self.args.style_loss * style_loss + self.args.ssim_loss * ssim_loss
            # self.args.att_loss 是 1，self.args.style_loss是0.025，
            # compute gradient and do SGD step
            total_loss.backward()

            self.model.step_all()

            # measure accuracy and record loss
            losses.update(l2_loss.item(), inputs.size(0))
            lossMask.update(att_loss.item(), inputs.size(0))
            lossWM.update(wm_loss.item(), inputs.size(0))

            if self.args.style_loss > 0 :
                lossvgg.update(style_loss.item(), inputs.size(0))

            if self.args.ssim_loss > 0 :
                lossssim.update(ssim_loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            suffix  = "({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss L2: {loss_label:.4f} | Loss Mask: {loss_mask:.4f} | loss WM: {loss_wm:.4f} | loss VGG: {loss_vgg:.4f} | loss SSIM: {loss_ssim:.4f}| loss MX: {loss_mx:.4f}".format(
                        batch=i + 1,
                        size=len(self.train_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss_label=losses.avg,
                        loss_mask=lossMask.avg,
                        loss_wm=lossWM.avg,
                        loss_vgg=lossvgg.avg,
                        loss_ssim=lossssim.avg,
                        loss_mx=lossMX.avg
                        )
            if current_index % 1000 == 0:
                print(suffix)

            if self.args.freq > 0 and current_index % self.args.freq == 0:
                self.validate(current_index)
                self.flush()
                self.save_checkpoint()

        self.record('train/loss_L2', losses.avg, epoch)
        self.record('train/loss_Mask', lossMask.avg, epoch)
        self.record('train/loss_WM', lossWM.avg, epoch)
        self.record('train/loss_VGG', lossvgg.avg, epoch)
        self.record('train/loss_SSIM', lossssim.avg, epoch)
        self.record('train/loss_MX', lossMX.avg, epoch)




    def validate(self, epoch):

        self.current_epoch = epoch
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        lossMask = AverageMeter()
        psnres = AverageMeter()
        ssimes = AverageMeter()
        psnres_int = AverageMeter()
        ssimes_int = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.val_loader))
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):

                current_index = len(self.val_loader) * epoch + i

                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)

                outputs = self.model(self.norm(inputs))
                imoutput,immask,imwatermark = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput

                imfinal = self.denorm(imoutput*immask + self.norm(inputs)*(1-immask))

                if i % 300 == 0:
                    # save the sample images，保存的是每一轮的结果，但是内存不够大了，因此就不保存了
                    ims = torch.cat([inputs,target,imfinal,immask.repeat(1,3,1,1)],dim=3)
                    # torchvision.utils.save_image(ims,os.path.join(self.args.checkpoint,'%s_%s.jpg'%(i,epoch)))

                # here two choice: mseLoss or NLLLoss
                imfinal_int = im_to_numpy(torch.clamp(imfinal[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                target_int = im_to_numpy(torch.clamp(target[0]*255,min=0.0,max=255.0)).astype(np.uint8)

                psnr = 10 * log10(1 / F.mse_loss(imfinal,target).item())
                ssim = pytorch_ssim.ssim(imfinal,target)
                psnr_int = compare_psnr(target_int,imfinal_int)
                ssim_int = compare_ssim(target_int,imfinal_int,multichannel=True)
                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))
                psnres_int.update(psnr_int, inputs.size(0))
                ssimes_int.update(ssim_int, inputs.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_L2: {loss_label:.4f} | Loss_Mask: {loss_mask:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}'.format(
                            batch=i + 1,
                            size=len(self.val_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss_label=losses.avg,
                            loss_mask=lossMask.avg,
                            psnr=psnres.avg,
                            ssim=ssimes.avg
                            )
                bar.next()
        bar.finish()
        
        print("Iter:%s,Losses:%s,Type-Float:(PSNR:%.4f,SSIM:%.4f), Type-Int:(PSNR:%.4f,SSIM:%.4f)"%(epoch, losses.avg,psnres.avg,ssimes.avg,psnres_int.avg,ssimes_int.avg))
        self.record('val/loss_L2', losses.avg, epoch)
        self.record('val/lossMask', lossMask.avg, epoch)
        self.record('val/PSNR', psnres.avg, epoch)
        self.record('val/SSIM', ssimes.avg, epoch)
        self.metric = psnres.avg

        self.model.train()
    # 通过vgg计算lpips
    def compute_lpips(img1, img2):
        # 将Int张量转换为浮点数张量，并将像素值缩放到[0, 1]范围内
        img1 = img1.float() / 255.0
        img2 = img2.float() / 255.0

        # 创建LPIPS模型
        lpips_model = lpips.LPIPS(net='vgg')

        # 计算LPIPS距离
        lpips_distance = lpips_model(img1, img2)

        return lpips_distance.item()
    def normforlpips(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        norm_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
        return norm_tensor

    def test(self):

        # switch to evaluate mode
        self.model.eval()
        print("==> testing VM model ")
        # 1为float类型,2为int类型
        ssimes_1 = AverageMeter()
        psnres_1 = AverageMeter()
        ssimes_2 = AverageMeter()
        psnres_2 = AverageMeter()
        ssimes_1_o = AverageMeter()
        psnres_1_o = AverageMeter()
        ssimes_2_o = AverageMeter()
        psnres_2_o = AverageMeter()
        import lpips
        lpips_model = lpips.LPIPS(net='alex')
        lpips_model = lpips_model.to(self.device)
        lpipses_1 = AverageMeter()
        lpipses_2 = AverageMeter()
        lpipses_1_o = AverageMeter()
        lpipses_2_o = AverageMeter()
        start_time = time.time()
        with torch.no_grad():
            for i, batches in enumerate(tqdm(self.val_loader)):

                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)
                # mask =batches['mask'].to(self.device)  # 其实没用到mask，我把它注释了

                # select the outputs by the giving arch
                outputs = self.model(self.norm(inputs))
                imoutput,immask,imwatermark = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                # 得到最终的重建图像
                imfinal = self.denorm(imoutput*immask + self.norm(inputs)*(1-immask))
                # 以下为防止出现找不到的错误
                psnr_1_o = 0
                ssim_1_o = 0
                lpips_1_o = 0
                psnr_2_o = 0
                ssim_2_o = 0
                # 计算未转化为int类型时的图像指标
                # 计算原始图像与带水印图像的图像指标，如果测试的不是CLWD或者其他数据集
                psnr_1_o = 10 * log10(1 / F.mse_loss(inputs,target).item())
                ssim_1_o = pytorch_ssim.ssim(inputs,target)
                lpips_1_o = lpips_model(inputs, target)
                # 计算原始图像与重建图像的图像指标
                psnr_1 = 10 * log10(1 / F.mse_loss(imfinal,target).item())
                ssim_1 = pytorch_ssim.ssim(imfinal,target)
                lpips_1 = lpips_model(imfinal, target)
                # recover the image to 255
                # 将图像重建为int类型,因为只有int类型可以保存图像
                imfinal_int = im_to_numpy(torch.clamp(imfinal[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                target_int = im_to_numpy(torch.clamp(target[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                inputs_int = im_to_numpy(torch.clamp(inputs[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                imoutput_int = im_to_numpy(torch.clamp(imoutput[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                # imwatermark_int = im_to_numpy(torch.clamp(imwatermark * 255, min=0.0, max=255.0)).astype(np.uint8)
                immask_int = immask[0][0].clamp(0.0, 1.0) * 255
                immask_int = immask_int.cpu().numpy().astype(np.uint8)
                imwatermark_int = im_to_numpy(torch.clamp(imwatermark[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                # immask_int = im_to_numpy(torch.clamp(immask[0][0]*255,min=0.0,max=255.0)).astype(np.uint8)
                # immask_int = (immask[0][0].clamp(0.0, 1.0) * 255).cpu().numpy().astype(np.uint8)
                # 保存图像的mask，原始模型输出
                immask_int = immask[0][0].clamp(0.0, 1.0) * 255
                immask_int = immask_int.cpu().numpy().astype(np.uint8)
                # imoutput为在S2AM之后的输出结果
                imoutput_int = im_to_numpy(torch.clamp(imoutput[0] * 255, min=0.0, max=255.0)).astype(np.uint8)

                # 储存图像,如果没有目录便创建,只保存_vx_前面那些字符
                # self.args.checkpoint.split("_vx")[0] 为存储位置
                # batches['name'][0]储存名
                # imfinal_int为储存结果
                dir = self.args.checkpoint.split("_vx")[0]
                if not os.path.exists(dir):
                    os.makedirs(dir)
                # 内存不够，只能注释掉下面三行，只保存view了
                # target_dir = os.path.join(dir, "host_image")
                # inputs_dir = os.path.join(dir, "watermarked_image")
                # result_dir = os.path.join(dir, "result")
                view_dir = os.path.join(dir, "view")
                # mask_dir = os.path.join(dir, "mask")
                # unmask_dir = os.path.join(dir, "unmask")
                # if not os.path.exists(target_dir):
                #     os.makedirs(target_dir)
                # if not os.path.exists(inputs_dir):
                #     os.makedirs(inputs_dir)
                # if not os.path.exists(mask_dir):
                #     os.makedirs(mask_dir)
                if not os.path.exists(view_dir):
                    os.makedirs(view_dir)
                # if not os.path.exists(unmask_dir):
                #     os.makedirs(unmask_dir)
                # 储存结果图，输入图，可视化图，目标图
                # skimage.io.imsave('%s/%s' % (result_dir, batches['name'][0]), imfinal_int)
                # skimage.io.imsave('%s/%s' % (inputs_dir, batches['name'][0]), inputs_int)
                # skimage.io.imsave('%s/%s' % (target_dir, batches['name'][0]), target_int)
                from PIL import Image
                view_image = Image.new('RGB', (1536, 256))
                view_image.paste(Image.fromarray(inputs_int), (0, 0))
                view_image.paste(Image.fromarray(target_int), (256, 0))
                view_image.paste(Image.fromarray(imfinal_int), (512, 0))
                view_image.paste(Image.fromarray(imoutput_int), (768, 0))
                view_image.paste(Image.fromarray(immask_int), (1024, 0))
                view_image.paste(Image.fromarray(imwatermark_int), (1280, 0))
                view_image.save('%s/%s' % (view_dir, batches['name'][0]))



                # 计算转化为int后的图像的结果，如果测试的不是CLWD数据集
                psnr_2_o = compare_psnr(target_int,inputs_int)
                ssim_2_o = compare_ssim(target_int,inputs_int,multichannel=True)
                psnr_2 = compare_psnr(target_int,imfinal_int)
                ssim_2 = compare_ssim(target_int,imfinal_int,multichannel=True)
                target_int = torch.tensor(target_int)
                target_int = target_int.permute(2,0,1).cuda()
                imfinal_int = torch.tensor(imfinal_int)
                imfinal_int = imfinal_int.permute(2,0,1).cuda()
                inputs_int = torch.tensor(inputs_int)
                inputs_int = inputs_int.permute(2,0,1).cuda()
                lpips_2 = lpips_model(self.normforlpips(target_int), self.normforlpips(imfinal_int))
                lpips_2_o = lpips_model(self.normforlpips(target_int), self.normforlpips(inputs_int))
                # 添加结果至数组中
                # 原本此处的1为inputs.size(0)
                psnres_1.update(psnr_1, 1)
                ssimes_1.update(ssim_1, 1)
                psnres_2.update(psnr_2, 1)
                ssimes_2.update(ssim_2, 1)
                psnres_1_o.update(psnr_1_o, 1)
                ssimes_1_o.update(ssim_1_o, 1)
                psnres_2_o.update(psnr_2_o, 1)
                ssimes_2_o.update(ssim_2_o, 1)
                lpipses_1.update(lpips_1, 1)
                lpipses_1_o.update(lpips_1_o, 1)
                lpipses_2.update(lpips_2, 1)
                lpipses_2_o.update(lpips_2_o, 1)
        end_time = time.time()
        total_time = end_time - start_time
        print("Total time", total_time, "s")
        print("fps", total_time/2025, "s")
        print("%s:Type-Float  PSNR:%.5f, PSNRO:%.5f, SSIM:%.5f, SSIMO:%.5f, LPIPS:%.5f, LPIPSO:%.5f"%(self.args.checkpoint, psnres_1.avg, psnres_1_o.avg, ssimes_1.avg, ssimes_1_o.avg, lpipses_1.avg, lpipses_1_o.avg))
        print("%s:Type-Int  PSNR:%.5f, PSNRO:%.5f, SSIM:%.5f, SSIMO:%.5f, LPIPS:%.5f, LPIPSO:%.5f"%(self.args.checkpoint, psnres_2.avg, psnres_2_o.avg, ssimes_2.avg, ssimes_2_o.avg, lpipses_2.avg, lpipses_2_o.avg))
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')  # 得到当前时间字符串
        start_index = self.args.checkpoint.find("/test/") + len("/test/")  # 找到 "/test/" 的索引，并加上长度获取下一个字符的起始索引
        end_index = self.args.checkpoint.find("/", start_index)  # 在起始索引之后继续查找下一个 "/" 的索引
        thismethod = self.args.checkpoint[start_index:end_index]  # 得到方法的名称
        start_index = self.args.checkpoint.find("/test/") + len("/test/")  # 找到 "/test/" 的索引，并加上长度获取下一个字符的起始索引
        first_index = self.args.checkpoint.find("/", start_index)  # 在起始索引之后继续查找第一个 "/" 的索引
        second_index = self.args.checkpoint.find("/", first_index + 1)  # 在第一个 "/" 索引之后继续查找第二个 "/" 的索引
        trainning_dataset = self.args.checkpoint[first_index + 1:second_index]  # 得到训练集名称
        ssimes_1_avg_float = float(ssimes_1.avg.item())
        ssimes_1_o_avg_float = float(ssimes_1_o.avg.item())
        lpipses_1_avg_float = float(lpipses_1.avg.item())
        lpipses_1_o_avg_float = float(lpipses_1_o.avg.item())
        float_data = {
            'time': current_time_str,
            'method': thismethod,
            'trainning dataset': trainning_dataset,
            'testing dataset': self.args.data,   # 测试数据集
            'PSNR': round(psnres_1.avg,3),
            'SSIM': round(ssimes_1_avg_float,4),
            'LPIPS': round(lpipses_1_avg_float,5),
            'PSNRO': round(psnres_1_o.avg,3),
            'SSIMO': round(ssimes_1_o_avg_float,4),
            'LPIPSO': round(lpipses_1_o_avg_float,5),
            'Improve-PSNR': round(psnres_1.avg-psnres_1_o.avg,4),
            'Improve-SSIM': round(ssimes_1_avg_float-ssimes_1_o_avg_float,5),
            'Improve-LPIPS': round(lpipses_1_o_avg_float-lpipses_1_avg_float,6)
        }
        print(float_data)
        lpipses_2_avg_float = float(lpipses_2.avg.item())
        lpipses_2_o_avg_float = float(lpipses_2_o.avg.item())
        int_data = {
            'time': current_time_str,
            'method': thismethod,
            'trainning dataset': trainning_dataset,
            'testing dataset': self.args.data,
            'PSNR': round(psnres_2.avg,3),
            'SSIM': round(ssimes_2.avg,4),
            'LPIPS': round(lpipses_2_avg_float,5),
            'PSNRO': round(psnres_2_o.avg,3),
            'SSIMO': round(ssimes_2_o.avg,4),
            'LPIPSO': round(lpipses_2_o_avg_float,5),
            'Improve-PSNR': round(psnres_2.avg-psnres_2_o.avg,4),
            'Improve-SSIM': round(ssimes_2.avg-ssimes_2_o.avg,5),
            'Improve-LPIPS': round(lpipses_2_o_avg_float-lpipses_2_avg_float,6)
        }
        print(int_data)
        floatfile = '/home/coolboy/huangwenhong/SplitNet_project/float_results.csv'
        intfile = '/home/coolboy/huangwenhong/SplitNet_project/int_results.csv'
        float_data_values = list(float_data.values())
        int_data_values = list(int_data.values())
        print("开始写入文件float_results.csv")
        with open(floatfile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(float_data_values)
        print("开始写入文件int_results.csv")
        with open(intfile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(int_data_values)
        print("写入完毕")
        # print("%s:PSNRO:%.5f, PSNRX:%.5f, PSNR:%.5f, SSIMO:%.5f, SSIMX:%.5f, SSIM:%.5f, LPIPSO:%.5f, LPIPSX:%.5f)"%(self.args.checkpoint,psnreso.avg, psnresx.avg,psnres.avg,ssimeso.avg,ssimesx.avg, ssimes.avg, lpipseso.avg, lpipsesx.avg))
        print("DONE.\n")