import csv
from datetime import datetime

import torch
import torch.nn as nn
from progress.bar import Bar
from tqdm import tqdm
from scripts.machines import pytorch_ssim
import json
import sys,time,os
import torchvision
from math import log10
import numpy as np
from .BasicModel import BasicModel
from evaluation import AverageMeter, compute_IoU, FScore, compute_RMSE
import torch.nn.functional as F
from src.utils.parallel import DataParallelModel, DataParallelCriterion
from src.utils.losses import VGGLoss, l1_relative,is_dic
from src.utils.imutils import im_to_numpy
import skimage.io
from skimage.measure import compare_psnr,compare_ssim
import torchvision
import pytorch_iou

class Losses(nn.Module):
    def __init__(self, argx, device, norm_func, denorm_func):
        super(Losses, self).__init__()
        self.args = argx
        self.masked_l1_loss, self.mask_loss = l1_relative, nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        if self.args.lambda_content > 0:
            self.vgg_loss = VGGLoss(self.args.sltype, style=self.args.lambda_style>0).to(device)
        
        if self.args.lambda_iou > 0:
            self.iou_loss = pytorch_iou.IOU(size_average=True)

        self.lambda_primary = self.args.lambda_primary
        self.gamma = 0.5
        self.norm = norm_func
        self.denorm = denorm_func

    def forward(self, synthesis, pred_ims, target, pred_ms, mask, threshold=0.5):
        pixel_loss, refine_loss, vgg_loss, mask_loss = [0]*4
        pred_ims = pred_ims if is_dic(pred_ims) else [pred_ims]
        
        # reconstruction loss
        pixel_loss += self.masked_l1_loss(pred_ims[-1], target, mask) # coarse stage
        if len(pred_ims) > 1:
            refine_loss = self.masked_l1_loss(pred_ims[0], target, mask) # refinement stage
        
        recov_imgs = [ self.denorm(pred_im*mask + (1-mask)*self.norm(target)) for pred_im in pred_ims ]        
        pixel_loss += sum([self.l1_loss(im,target) for im in recov_imgs]) * 1.5
        

        if self.args.lambda_content > 0:
            vgg_loss = [self.vgg_loss(im,target,mask) for im in recov_imgs]
            vgg_loss = sum([vgg['content'] for vgg in vgg_loss]) * self.args.lambda_content + \
                       sum([vgg['style'] for vgg in vgg_loss]) * self.args.lambda_style

        # mask loss
        # print(pred_ms.size())
        pred_ms = [F.interpolate(ms, size=mask.shape[2:], mode='bilinear') for ms in pred_ms]
        pred_ms = [pred_m.clamp(0,1) for pred_m in pred_ms]
        mask = mask.clamp(0,1)

        final_mask_loss = 0
        final_mask_loss += self.mask_loss(pred_ms[0], mask)
        
        primary_mask = pred_ms[1::2][::-1]   # 存mask时,有两个结果,一个放在奇数位置,论文的自己设计的mask放在偶数位置
        self_calibrated_mask = pred_ms[2::2][::-1]
        # primary prediction
        primary_loss =  sum([self.mask_loss(pred_m, mask) * (self.gamma**i) for i,pred_m in enumerate(primary_mask)])
        # self calibrated Branch
        self_calibrated_loss =  sum([self.mask_loss(pred_m, mask) * (self.gamma**i) for i,pred_m in enumerate(self_calibrated_mask)])
        if self.args.lambda_iou > 0:
            self_calibrated_loss += sum([self.iou_loss(pred_m, mask) * (self.gamma**i) for i,pred_m in enumerate(self_calibrated_mask)]) * self.args.lambda_iou

        mask_loss = final_mask_loss + self_calibrated_loss + self.lambda_primary * primary_loss
        return pixel_loss, refine_loss, vgg_loss, mask_loss




class SLBR(BasicModel):
    def __init__(self,**kwargs):
        BasicModel.__init__(self,**kwargs)
        self.loss = Losses(self.args, self.device, self.norm, self.denorm)
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        # self.model.set_optimizers()
        if self.args.resume != '':
            self.resume(self.args.resume)
       
    def train(self,epoch,lr):

        self.current_epoch = epoch

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_meter = AverageMeter()
        loss_mask_meter = AverageMeter()
        loss_vgg_meter = AverageMeter()
        loss_refine_meter = AverageMeter()
        f1_meter = AverageMeter()
        psnres = AverageMeter()
        # switch to train mode
        self.model.set_optimizers()
        self.model.train()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))
        for i, batches in enumerate(self.train_loader):
            current_index = len(self.train_loader) * epoch + i

            inputs = batches['image'].float().to(self.device)
            target = batches['target'].float().to(self.device)
            mask = batches['mask'].float().to(self.device)
            # wm =  batches['wm'].float().to(self.device)
            # alpha_gt = batches['alpha'].float().to(self.device)
            # img_path = batches['img_path']
            # from thop import clever_format
            # from thop import profile
            # flops, params = profile(self.model, inputs=(self.norm(inputs),), verbose=False)
            # flops, params = clever_format([flops, params], "%.3f")
            # print(flops)
            # print(params)
            outputs = self.model(self.norm(inputs))
            self.model.zero_grad_all()
            coarse_loss, refine_loss, style_loss, mask_loss = self.loss(
                inputs,outputs[0],self.norm(target),outputs[1],mask)
            
            total_loss = self.args.lambda_l1*(coarse_loss+refine_loss) + self.args.lambda_mask * (mask_loss)  + style_loss
            imoutput, immask, imwatermark = outputs
            immask = immask[0]
            if len(imoutput) > 1:
                imcoarse = imoutput[1]
                imcoarse = imcoarse * immask + inputs * (1 - immask)
            else:
                imcoarse = None
            imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            imfinal = self.denorm(imoutput * immask + self.norm(inputs) * (1 - immask))
            psnr = 10 * log10(1 / F.mse_loss(imfinal, target).item())
            psnres.update(psnr, inputs.size(0))
            # compute gradient and do SGD step
            total_loss.backward()
            self.model.step_all()

            # measure accuracy and record loss
            losses_meter.update(coarse_loss.item(), inputs.size(0))
            loss_mask_meter.update(mask_loss.item(), inputs.size(0))
            if isinstance(refine_loss,int):
                loss_refine_meter.update(refine_loss, inputs.size(0))
            else:
                loss_refine_meter.update(refine_loss.item(), inputs.size(0))
            
            f1 = FScore(outputs[1][0], mask).item()
            f1_meter.update(f1, inputs.size(0))
            if self.args.lambda_content > 0  and not isinstance(style_loss,int):
                loss_vgg_meter.update(style_loss.item(), inputs.size(0))

            # measure elapsed timec
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            suffix  = "({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | loss L1: {loss_label:.4f} | loss Refine: {loss_refine:.4f} | loss VGG: {loss_vgg:.4f} | loss Mask: {loss_mask:.4f} | mask F1: {mask_f1:.4f} | PSNR: {psnr:.4f} ".format(
                        batch=i + 1,
                        size=len(self.train_loader),
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        loss_label=losses_meter.avg,
                        loss_refine=loss_refine_meter.avg,
                        loss_vgg=loss_vgg_meter.avg,
                        loss_mask=loss_mask_meter.avg,
                        mask_f1=f1_meter.avg,
                        psnr = psnres.avg
                        )
            if current_index % 100 == 0:
                print(suffix)

            if self.args.freq > 0 and current_index % self.args.freq == 0:
                self.validate(current_index)
                self.flush()
                self.save_checkpoint()
            if i % 100 == 0:
                self.record('train/loss_L2', losses_meter.avg, current_index)
                self.record('train/loss_Refine', loss_refine_meter.avg, current_index)
                self.record('train/loss_VGG', loss_vgg_meter.avg, current_index)
                self.record('train/loss_Mask', loss_mask_meter.avg, current_index)
                self.record('train/mask_F1', f1_meter.avg, current_index)

                mask_pred = outputs[1][0]
                bg_pred = self.denorm(outputs[0][0]*mask_pred + (1-mask_pred)*self.norm(inputs))
                show_size = 5 if inputs.shape[0] > 5 else inputs.shape[0]
                self.image_display = torch.cat([
                    inputs[0:show_size].detach().cpu(),             # input image
                    target[0:show_size].detach().cpu(),                        # ground truth
                    bg_pred[0:show_size].detach().cpu(),       # refine out
                    mask[0:show_size].detach().cpu().repeat(1,3,1,1),
                    outputs[1][0][0:show_size].detach().cpu().repeat(1,3,1,1),
                    outputs[1][-2][0:show_size].detach().cpu().repeat(1,3,1,1)
                ],dim=0)
                image_dis = torchvision.utils.make_grid(self.image_display, nrow=show_size)
                self.writer.add_image('Image', image_dis, current_index)
            del outputs


    def validate(self, epoch):

        self.current_epoch = epoch
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_meter = AverageMeter()
        loss_mask_meter = AverageMeter()
        psnr_meter = AverageMeter()
        fpsnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        rmse_meter = AverageMeter()
        rmsew_meter = AverageMeter()
        

        coarse_psnr_meter = AverageMeter()
        coarse_rmsew_meter = AverageMeter()
        psnres_int = AverageMeter()
        ssimes_int = AverageMeter()
        iou_meter = AverageMeter()
        f1_meter = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.val_loader))
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):
                
                current_index = len(self.val_loader) * epoch + i

                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)
                mask = batches['mask'].to(self.device)
                # alpha_gt = batches['alpha'].float().to(self.device)

                outputs = self.model(self.norm(inputs))
                imoutput,immask,imwatermark = outputs
                
                immask = immask[0]
                if len(imoutput) > 1:
                    imcoarse = imoutput[1]
                    imcoarse = imcoarse*immask + inputs*(1-immask)
                else: imcoarse = None
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput

                imfinal = self.denorm(imoutput*immask + self.norm(inputs)*(1-immask))

                eps = 1e-6
                psnr = 10 * log10(1 / F.mse_loss(imfinal,target).item()) 
                fmse = F.mse_loss(imfinal*mask, target*mask, reduction='none').sum(dim=[1,2,3]) / (mask.sum(dim=[1,2,3])*3+eps)
                fpsnr = 10 * torch.log10(1 / fmse).mean().item()
                ssim = pytorch_ssim.ssim(imfinal,target)
                if imcoarse is not None:
                    psnr_coarse = 10 * log10(1 / F.mse_loss(imcoarse,target).item())  
                    rmsew_coarse = compute_RMSE(imcoarse, target, mask, is_w=True)
                    coarse_psnr_meter.update(psnr_coarse, inputs.size(0))
                    coarse_rmsew_meter.update(rmsew_coarse, inputs.size(0))

                psnr_meter.update(psnr, inputs.size(0))
                fpsnr_meter.update(fpsnr, inputs.size(0))
                ssim_meter.update(ssim, inputs.size(0))
                rmse_meter.update(compute_RMSE(imfinal,target,mask),inputs.size(0))
                rmsew_meter.update(compute_RMSE(imfinal,target,mask,is_w=True), inputs.size(0))

                iou = compute_IoU(immask, mask)
                iou_meter.update(iou, inputs.size(0))
                f1 = FScore(immask, mask).item()
                f1_meter.update(f1, inputs.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                imfinal_int = im_to_numpy(torch.clamp(imfinal[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                target_int = im_to_numpy(torch.clamp(target[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                psnr_int = compare_psnr(target_int,imfinal_int)
                ssim_int = compare_ssim(target_int,imfinal_int,multichannel=True)
                psnres_int.update(psnr_int, inputs.size(0))
                ssimes_int.update(ssim_int, inputs.size(0))
                # plot progress
                if imcoarse is None:
                    bar.suffix  = '({batch}/{size})  PSNR: {psnr:.4f} | fPSNR: {fpsnr:.4f} | SSIM: {ssim:.4f} | RMSE: {rmse:.4f} | RMSEw: {rmsew:.4f} | IoU: {iou:.4f} | F1: {f1:.4f}'.format(
                                batch=i + 1,
                                size=len(self.val_loader),
                                psnr=psnr_meter.avg,
                                fpsnr=fpsnr_meter.avg,
                                ssim=ssim_meter.avg,
                                rmse=rmse_meter.avg,
                                rmsew=rmsew_meter.avg,
                                iou=iou_meter.avg,
                                f1=f1_meter.avg
                                )
                else:
                    bar.suffix  = '({batch}/{size}) |CPSNR:{cpsnr:.4f}|CRMSEw:{crmsew:.4f}|PSNR:{psnr:.4f}|fPSNR:{fpsnr:.4f}|RMSE:{rmse:.4f}|RMSEw:{rmsew:.4f}|SSIM:{ssim:.4f}|F1:{f1:.4f}'.format(
                            batch=i + 1,
                            size=len(self.val_loader),
                            cpsnr=coarse_psnr_meter.avg,
                            crmsew=coarse_rmsew_meter.avg,
                            psnr=psnr_meter.avg,
                            fpsnr=fpsnr_meter.avg,
                            ssim=ssim_meter.avg,
                            rmse=rmse_meter.avg,
                            rmsew=rmsew_meter.avg,
                            f1=f1_meter.avg
                            )
                # else:
                #     suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | CPSNR: {cpsnr:.4f} | CRMSEw: {crmsew:.4f} | PSNR: {psnr:.4f} | fPSNR: {fpsnr:.4f} | RMSE: {rmse:.4f} | RMSEw: {rmsew:.4f} | SSIM: {ssim:.4f} | IoU: {iou:.4f} | F1: {f1:.4f}'.format(
                #             batch=i + 1,
                #             size=len(self.val_loader),
                #             data=data_time.val,
                #             bt=batch_time.val,
                #             total=bar.elapsed_td,
                #             eta=bar.eta_td,
                #             cpsnr=coarse_psnr_meter.avg,
                #             crmsew=coarse_rmsew_meter.avg,
                #             psnr=psnr_meter.avg,
                #             fpsnr=fpsnr_meter.avg,
                #             ssim=ssim_meter.avg,
                #             rmse=rmse_meter.avg,
                #             rmsew=rmsew_meter.avg,
                #             iou=iou_meter.avg,
                #             f1=f1_meter.avg
                #             )
                bar.next()

        print("Total:")
        bar.finish()
        print("Iter:%s,Losses:%s,Type-Float:(PSNR:%.4f,SSIM:%.4f), Type-Int:(PSNR:%.4f,SSIM:%.4f)" % (
        epoch, losses_meter.avg, psnr_meter.avg, ssim_meter.avg, psnres_int.avg, ssimes_int.avg))
        # print("Iter:%s,losses:%s,PSNR:%.4f,SSIM:%.4f"%(epoch, losses_meter.avg,psnr_meter.avg,ssim_meter.avg))
        self.record('val/loss_L2', losses_meter.avg, epoch)
        self.record('val/loss_mask', loss_mask_meter.avg, epoch)
        self.record('val/PSNR', psnr_meter.avg, epoch)
        self.record('val/SSIM', ssim_meter.avg, epoch)
        self.record('val/RMSEw', rmsew_meter.avg, epoch)
        self.metric = psnr_meter.avg

        self.model.train()
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
        lpips_model = lpips.LPIPS(net='vgg')
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
                imoutput,immask_all,imwatermark = outputs
                # imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                # 得到最终的重建图像
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                # imwatermark = imwatermark[0] if is_dic(imwatermark) else imwatermark

                immask = immask_all[0]

                imfinal = imoutput * immask + self.norm(inputs) * (1 - immask)

                # imfinal = self.denorm(imoutput*immask + self.norm(inputs)*(1-immask))
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
                # if not os.path.exists(target_dir):
                #     os.makedirs(target_dir)
                # if not os.path.exists(inputs_dir):
                #     os.makedirs(inputs_dir)
                # if not os.path.exists(result_dir):
                #     os.makedirs(result_dir)
                if not os.path.exists(view_dir):
                    os.makedirs(view_dir)
                # 储存结果图，输入图，可视化图，目标图
                # skimage.io.imsave('%s/%s' % (result_dir, batches['name'][0]), imfinal_int)
                # skimage.io.imsave('%s/%s' % (inputs_dir, batches['name'][0]), inputs_int)
                # skimage.io.imsave('%s/%s' % (target_dir, batches['name'][0]), target_int)
                from PIL import Image
                view_image = Image.new('RGB', (1280, 256))
                view_image.paste(Image.fromarray(inputs_int), (0, 0))
                view_image.paste(Image.fromarray(target_int), (256, 0))
                view_image.paste(Image.fromarray(imfinal_int), (512, 0))
                view_image.paste(Image.fromarray(imoutput_int), (768, 0))
                view_image.paste(Image.fromarray(immask_int), (1024, 0))
                # view_image.paste(Image.fromarray(imwatermark_int), (1280, 0))
                view_image.save('%s/%s' % (view_dir, batches['name'][0]))

                # 计算转化为int后的图像的结果，如果测试的不是CLWD数据集
                psnr_2_o = compare_psnr(target_int,inputs_int)
                ssim_2_o = compare_ssim(target_int,inputs_int,multichannel=True)
                psnr_2 = compare_psnr(target_int,imfinal_int)
                ssim_2 = compare_ssim(target_int,imfinal_int,multichannel=True)
                target_float = torch.from_numpy(target_int).float() / 255.0
                imfinal_float = torch.from_numpy(imfinal_int).float() / 255.0
                inputs_float = torch.from_numpy(inputs_int).float() / 255.0
                target_float = target_float.permute(2, 0, 1).unsqueeze(0).to(self.device)
                imfinal_float = imfinal_float.permute(2, 0, 1).unsqueeze(0).to(self.device)
                inputs_float = inputs_float.permute(2, 0, 1).unsqueeze(0).to(self.device)
                lpips_2 = lpips_model(target_float, imfinal_float)
                lpips_2_o = lpips_model(target_float, inputs_float)
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
        start_index = self.args.checkpoint.find("/SLBRtest/") + len("/SLBRtest/")  # 找到 "/test/" 的索引，并加上长度获取下一个字符的起始索引
        end_index = self.args.checkpoint.find("/", start_index)  # 在起始索引之后继续查找下一个 "/" 的索引
        thismethod = self.args.checkpoint[start_index:end_index]  # 得到方法的名称
        start_index = self.args.checkpoint.find("/SLBRtest/") + len("/SLBRtest/")  # 找到 "/test/" 的索引，并加上长度获取下一个字符的起始索引
        first_index = self.args.checkpoint.find("/", start_index)  # 在起始索引之后继续查找第一个 "/" 的索引
        second_index = self.args.checkpoint.find("/", first_index + 1)  # 在第一个 "/" 索引之后继续查找第二个 "/" 的索引
        trainning_dataset = self.args.checkpoint[first_index + 1:second_index]  # 得到训练集名称
        slbr_name = "-SLBR"
        thismethod = thismethod + slbr_name
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

   