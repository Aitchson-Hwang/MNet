# Machine to train MNet (Only used to train, use TestAllMethods.py when testing)

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
import shutil
class Losses(nn.Module):
    def __init__(self, argx, device, norm_func, denorm_func):
        super(Losses, self).__init__()
        self.args = argx
        self.masked_l1_loss = l1_relative
        self.l1_loss = nn.L1Loss()
        self.mask_loss = nn.BCELoss()
        self.iou_loss = pytorch_iou.IOU(size_average=True)
        if self.args.lambda_content > 0:
            self.vgg_loss = VGGLoss(self.args.sltype, style=self.args.lambda_style>0).to(device)


        self.gamma = 0.5
        self.norm = norm_func
        self.denorm = denorm_func

    def dice_loss(self, pred, target):
        smooth = 1e-5
        intersection = torch.sum(pred * target, dim=(1, 2, 3))
        union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss

    def forward(self, synthesis, pred_ims, target, pred_ms, mask, threshold=0.5):
        pixel_loss, vgg_loss = [0]*2
        pred_ims = pred_ims if is_dic(pred_ims) else [pred_ims]
        
        # reconstruction loss
        pixel_loss += self.masked_l1_loss(pred_ims[-1], target, mask)
        
        recov_imgs = [ self.denorm(pred_im*mask + (1-mask)*self.norm(target)) for pred_im in pred_ims ]        
        pixel_loss += sum([self.l1_loss(im,target) for im in recov_imgs]) * 1.5
        
        # VGG Loss
        if self.args.lambda_content > 0:
            vgg_loss = [self.vgg_loss(im,target,mask) for im in recov_imgs]
            vgg_loss = sum([vgg['content'] for vgg in vgg_loss]) * self.args.lambda_content + \
                       sum([vgg['style'] for vgg in vgg_loss]) * self.args.lambda_style

        bce_mask_loss, dice_mask_loss, iou_mask_loss = [0] * 3
        pred_ms = pred_ms.clamp(0, 1)
        mask = mask.clamp(0, 1)
        # 计算BCELoss
        final_mask_loss = 0
        final_mask_loss += self.mask_loss(pred_ms, mask)
        dice_loss = 0
        if self.args.dice == 1:
            dice_loss += self.dice_loss(pred_ms, mask)
        iou_loss = 0
        iou_loss += self.iou_loss(pred_ms, mask)
        bce_mask_loss = final_mask_loss
        dice_mask_loss = dice_loss
        iou_mask_loss = iou_loss
        # mask_loss = bce_mask_loss * 0.5 + dice_mask_loss * 0.25 + iou_mask_loss * 0.5
        mask_loss = bce_mask_loss * 0.5 + dice_mask_loss + iou_mask_loss
        # mask_loss = bce_mask_loss * 0.5 + iou_mask_loss * 0.5
        return pixel_loss, vgg_loss, mask_loss


class SUNet(BasicModel):
    def __init__(self,**kwargs):
        BasicModel.__init__(self,**kwargs)
        self.loss = Losses(self.args, self.device, self.norm, self.denorm)
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        # self.clip_value = 1.0   # 为防止梯度爆炸，而进行梯度裁剪
        if self.args.resume != '':
            self.resume(self.args.resume)
        if self.args.resume2 != '':
            self.resume2(self.args.resume2)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self,epoch,lr):
        self.current_epoch = epoch
        self.model.set_optimizers(lr)
        if self.args.arch2 != 'None':
            self.model2.set_optimizers(lr)
        batch_time = AverageMeter()
        losses_meter = AverageMeter()
        loss_vgg_meter = AverageMeter()
        loss_mask_meter = AverageMeter()
        # switch to train mode
        self.model.train()
        psnres = AverageMeter()
        f1s = AverageMeter()
        if self.args.is_clip == 1:
            self.clip_value = 1.0   # 为防止梯度爆炸
        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))
        total_params = self.count_parameters(self.model)
        print(f"Total number of trainable parameters: {total_params / 1000000.0} M")
        for i, batches in enumerate(self.train_loader):
            current_index = len(self.train_loader) * epoch + i

            inputs = batches['image'].float().to(self.device)
            target = batches['target'].float().to(self.device)
            mask = batches['mask'].float().to(self.device)
            outputs = self.model(self.norm(inputs))
            imoutput = outputs[0]
            immask = outputs[1]
            self.model.zero_grad_all()
            coarse_loss, style_loss, mask_loss = self.loss(
                inputs, imoutput, self.norm(target), immask, mask)
            total_loss = self.args.lambda_l1 * (coarse_loss) * 2 + style_loss * 0.5 + mask_loss
            total_loss.backward()
            if self.args.is_clip == 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)  # 为防止梯度爆炸
            self.model.step_all()
            if self.args.arch2 != 'None':
                self.model2.step_all()
                if self.args.is_clip == 1:
                    torch.nn.utils.clip_grad_norm_(self.model2.parameters(), self.clip_value)  # 为防止梯度爆炸

            imfinal = self.denorm(imoutput * mask + self.norm(inputs) * (1 - mask))
            psnr = 10 * log10(1 / F.mse_loss(imfinal, target).item())
            psnres.update(psnr, inputs.size(0))

            f1 = FScore(immask, mask).item()
            f1s.update(f1, inputs.size(0))

            # measure accuracy and record loss
            losses_meter.update(coarse_loss.item(), inputs.size(0))

            if self.args.lambda_content > 0  and not isinstance(style_loss,int):
                loss_vgg_meter.update(style_loss.item(), inputs.size(0))
            loss_mask_meter.update(mask_loss.item(), inputs.size(0))
            # measure elapsed timec
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            suffix  = "({batch}/{size}) Batch:{bt:.3f}s | Total: {total:} | loss L1: {loss_label:.4f} | loss VGG: {loss_vgg:.4f} | PSNR: {psnr:.4f} | loss Mask: {loss_mask:.4f} | F1: {f1_score:.4f}".format(
                        batch=i + 1,
                        size=len(self.train_loader),
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        loss_label=losses_meter.avg,
                        loss_vgg=loss_vgg_meter.avg,
                        psnr=psnres.avg,
                        loss_mask = loss_mask_meter.avg,
                        f1_score = f1s.avg
                        )
            if current_index % 100 == 0:
                print(suffix)

            if self.args.freq > 0 and current_index % self.args.freq == 0:
                self.validate(current_index)
                self.flush()
                self.save_checkpoint()
            if i % 100 == 0:
                self.record('train/loss_L2', losses_meter.avg, current_index)
                self.record('train/loss_VGG', loss_vgg_meter.avg, current_index)
                self.record('train/loss_Mask', loss_mask_meter.avg, current_index)

            del outputs


    def validate(self, epoch):

        self.current_epoch = epoch
        
        batch_time = AverageMeter()
        losses_meter = AverageMeter()
        loss_mask_meter = AverageMeter()
        ssim_meter = AverageMeter()
        psnrs = AverageMeter()
        f1_meter = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        if self.args.arch2 != 'None':
            self.model2.eval()
        import lpips
        lpips_model = lpips.LPIPS(net='alex')
        lpips_model = lpips_model.to(self.device)
        lpipses_1 = AverageMeter()
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
                imoutput = outputs[0]
                immask = outputs[1]
                imfinal = self.denorm(imoutput * immask + self.norm(inputs) * (1 - immask))

                if self.args.arch2 != 'None':
                    outputs2 = self.model2(self.norm(imfinal))
                    imoutput = outputs2[0]
                    imfinal = self.denorm(imoutput * immask + self.norm(inputs) * (1 - immask))

                imfinal_int = im_to_numpy(torch.clamp(imfinal[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                target_int = im_to_numpy(torch.clamp(target[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                # float
                psnr_1 = 10 * log10(1 / F.mse_loss(imfinal, target).item())
                ssim_1 = pytorch_ssim.ssim(imfinal, target)
                lpips_1 = lpips_model(self.normforlpips(imfinal), self.normforlpips(target))

                f1 = FScore(immask, mask).item()
                f1_meter.update(f1, inputs.size(0))
                # int
                ssim_2 = compare_ssim(target_int, imfinal_int, multichannel=True)
                psnr_int = compare_psnr(target_int, imfinal_int)

                psnrs.update(psnr_1, inputs.size(0))
                ssim_meter.update(ssim_1, inputs.size(0))
                lpipses_1.update(lpips_1, inputs.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) | PSNR: {psnr:.4f} | F1: {f1_score:.4f}  '.format(
                            batch=i + 1,
                            size=len(self.val_loader),
                            psnr=psnrs.avg,
                            f1_score=f1_meter.avg
                            )
                bar.next()
        print("Total:")
        bar.finish()
        print("Iter:%s,losses:%s, PSNR:%.4f, F1:%.4f, SSIM:%.4f, LPIPS:%.4f" % (epoch, losses_meter.avg, psnrs.avg, f1_meter.avg,ssim_meter.avg,lpipses_1.avg))
        self.record('val/loss_L2', losses_meter.avg, epoch)
        self.record('val/loss_mask', loss_mask_meter.avg, epoch)
        self.record('val/PSNR', psnrs.avg, epoch)
        self.record('val/SSIM', ssim_meter.avg, epoch)
        self.record('val/F1', f1_meter.avg, epoch)
        self.metric = psnrs.avg
        self.metric2 = ssim_meter.avg
        self.metric3 = lpipses_1.avg
        self.model.train()
    def normforlpips(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        norm_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
        return norm_tensor

    def save_checkpoint(self, filename='checkpoint.pth.tar', snapshot=None):
        # is_best = True if self.best_acc < self.metric else False
        is_best = False
        if self.best_acc < self.metric:
            is_best = True
        elif self.best_acc == self.metric:
            if self.best_acc2 < self.metric2:
                is_best = True
            elif self.best_acc2 == self.metric2:
                if self.best_acc3 < self.metric3:
                    is_best = True

        if is_best:
            self.best_acc = self.metric
            self.best_acc2 = self.metric2
            self.best_acc3 = self.metric3

        state = {
            'epoch': self.current_epoch + 1,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
        }

        filepath = os.path.join(self.args.checkpoint, filename)
        torch.save(state, filepath)

        if snapshot and state['epoch'] % snapshot == 0:
            shutil.copyfile(filepath, os.path.join(self.args.checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))

        if is_best:
            self.best_acc = self.metric
            print('Saving Best Metric with PSNR:%s' % self.best_acc)
            if not os.path.exists(self.args.checkpoint): os.makedirs(self.args.checkpoint)
            shutil.copyfile(filepath, os.path.join(self.args.checkpoint, 'model_best.pth.tar'))

        if self.args.arch2 != 'None':
            state2 = {
                'epoch': self.current_epoch + 1,
                'arch': self.args.arch2,
                'state_dict': self.model2.state_dict(),
                'best_acc': self.best_acc2,
                'optimizer': self.optimizer2.state_dict() if self.optimizer2 else None,
            }
            filepath2 = os.path.join(self.args.checkpoint2, filename)
            torch.save(state2, filepath2)
            if snapshot and state2['epoch'] % snapshot == 0:
                shutil.copyfile(filepath2,
                                os.path.join(self.args.checkpoint2, 'checkpoint_{}.pth.tar'.format(state2.epoch)))
            if is_best:
                if not os.path.exists(self.args.checkpoint2): os.makedirs(self.args.checkpoint2)
                shutil.copyfile(filepath2, os.path.join(self.args.checkpoint2, 'model_best.pth.tar'))
                print("successfully saving arch2")


    def resume2(self, resume_path):
        # if isfile(resume_path):
        if not os.path.exists(resume_path):
            resume_path = os.path.join(self.args.checkpoint2, 'checkpoint.pth.tar')
        if not os.path.exists(resume_path):
            raise Exception("=> no checkpoint found at '{}'".format(resume_path))

        print("=> loading checkpoint '{}'".format(resume_path))
        current_checkpoint = torch.load(resume_path)
        if isinstance(current_checkpoint['state_dict'], torch.nn.DataParallel):
            current_checkpoint['state_dict'] = current_checkpoint['state_dict'].module

        if isinstance(current_checkpoint['optimizer'], torch.nn.DataParallel):
            current_checkpoint['optimizer'] = current_checkpoint['optimizer'].module

        if self.args.start_epoch == 0:
            self.args.start_epoch = current_checkpoint['epoch']
        self.metric2 = current_checkpoint['best_acc']
        items = list(current_checkpoint['state_dict'].keys())

        ## restore the learning rate
        lr = self.args.lr
        for epoch in self.args.schedule:
            if epoch <= self.args.start_epoch:
                lr *= self.args.gamma
        optimizers = [getattr(self.model, attr) for attr in dir(self.model) if
                      attr.startswith("optimizer") and getattr(self.model, attr) is not None]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # ---------------- Load Model Weights --------------------------------------
        self.model2.load_state_dict(current_checkpoint['state_dict'], strict=True)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_path, current_checkpoint['epoch']))

   