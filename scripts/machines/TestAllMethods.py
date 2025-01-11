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
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class AllMethods(BasicModel):
    def __init__(self,**kwargs):
        BasicModel.__init__(self,**kwargs)
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        if self.args.resume != '':
            self.resume(self.args.resume)
    def normforlpips(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        norm_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
        return norm_tensor

    def test(self):
        # switch to evaluate mode
        self.model.eval()
        print("==> testing models ")
        # 1 is Type float, 2 is Type int
        ssimes_1 = AverageMeter()
        psnres_1 = AverageMeter()
        ssimes_2 = AverageMeter()
        psnres_2 = AverageMeter()
        ssimes_1_o = AverageMeter()
        psnres_1_o = AverageMeter()
        ssimes_2_o = AverageMeter()
        psnres_2_o = AverageMeter()
        # Calculate LPIPS
        import lpips
        lpips_model = lpips.LPIPS(net='squeeze')
        lpips_model = lpips_model.to(self.device)
        lpipses_1 = AverageMeter()
        lpipses_2 = AverageMeter()
        lpipses_1_o = AverageMeter()
        lpipses_2_o = AverageMeter()
        # Evaluate the mask prediction
        iou_meter = AverageMeter()
        f1_meter  = AverageMeter()
        # Evaluate the model time to precessed one image
        start_time = time.time()
        with torch.no_grad():
            for i, batches in enumerate(tqdm(self.val_loader)):
                if self.args.is_realimg == 1:
                    inputs = batches['image'].to(self.device)
                    target = batches['target'].to(self.device)
                    outputs = self.model(self.norm(inputs))
                    if self.args.arch == 'vm3' or self.args.arch == 'vvv4n':
                        # if BVMR or SplitNet
                        imoutput, immask, imwatermark = outputs
                        imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                    elif self.args.arch == 'slbr':
                        imoutput, immask, imwatermark = outputs
                        immask = immask[0]
                        imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                    elif self.args.arch == 'denet':
                        imoutput, immask_all, _, _ = outputs
                        imcoarse = imoutput[1]
                        imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                        immask = immask_all[0]
                    else:
                        imoutput, immask = outputs
                    imfinal = imoutput * immask + self.norm(inputs) * (1 - immask)
                    # 将图像重建为int类型,因为只有int类型可以保存图像
                    imfinal_int = im_to_numpy(torch.clamp(imfinal[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                    target_int = im_to_numpy(torch.clamp(target[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                    inputs_int = im_to_numpy(torch.clamp(inputs[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                    imoutput_int = im_to_numpy(torch.clamp(imoutput[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                    immask_int = immask[0][0].clamp(0.0, 1.0) * 255
                    immask_int = immask_int.cpu().numpy().astype(np.uint8)
                    dir = self.args.checkpoint
                    if self.args.simage == 1:
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                        view_dir = os.path.join(dir, "view")
                        if not os.path.exists(view_dir):
                            os.makedirs(view_dir)
                        from PIL import Image
                        view_image = Image.new('RGB', (1280, 256))
                        view_image.paste(Image.fromarray(inputs_int), (0, 0))
                        view_image.paste(Image.fromarray(target_int), (256, 0))
                        view_image.paste(Image.fromarray(imfinal_int), (512, 0))
                        view_image.paste(Image.fromarray(imoutput_int), (768, 0))
                        # view_image.paste(Image.fromarray(imcoarsef_int), (1024, 0))
                        # view_image.paste(Image.fromarray(imcoarse_int), (1280, 0))
                        view_image.paste(Image.fromarray(immask_int), (1024, 0))
                        view_image.save('%s/%s' % (view_dir, batches['name'][0]))

                        continue
                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)
                mask = batches['mask'].to(self.device)
                # select the outputs by the giving arch
                outputs = self.model(self.norm(inputs))
                if self.args.arch == 'vm3' or self.args.arch == 'vvv4n':
                    # if BVMR or SplitNet
                    imoutput, immask, imwatermark = outputs
                    imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                elif self.args.arch == 'slbr':
                    imoutput, immask, imwatermark = outputs
                    immask = immask[0]
                    imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                elif self.args.arch == 'denet':
                    imoutput, immask_all, _, _ = outputs
                    imcoarse = imoutput[1]
                    imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                    immask = immask_all[0]
                else:
                    imoutput, immask = outputs
                imfinal = imoutput * immask + self.norm(inputs) * (1 - immask)
                # imcoarsef = imcoarse * immask + self.norm(inputs) * (1 - immask)
                iou = compute_IoU(immask, mask)
                iou_meter.update(iou, inputs.size(0))
                f1 = FScore(immask, mask).item()
                f1_meter.update(f1, inputs.size(0))
                gtmask_int = mask[0][0].clamp(0.0, 1.0) * 255
                gtmask_int = gtmask_int.cpu().numpy().astype(np.uint8)
                # 以下为防止出现找不到的错误
                psnr_1_o = 0
                ssim_1_o = 0
                lpips_1_o = 0
                psnr_2_o = 0
                ssim_2_o = 0
                # 计算未转化为int类型时的图像指标
                psnr_1_o = 10 * log10(1 / F.mse_loss(inputs,target).item())
                ssim_1_o = pytorch_ssim.ssim(inputs,target)
                lpips_1_o = lpips_model(self.normforlpips(inputs), self.normforlpips(target))

                # 计算原始图像与重建图像的图像指标
                psnr_1 = 10 * log10(1 / F.mse_loss(imfinal,target).item())
                ssim_1 = pytorch_ssim.ssim(imfinal,target)
                lpips_1 = lpips_model(self.normforlpips(imfinal), self.normforlpips(target))
                # recover the image to 255
                # 将图像重建为int类型,因为只有int类型可以保存图像
                imfinal_int = im_to_numpy(torch.clamp(imfinal[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                target_int = im_to_numpy(torch.clamp(target[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                inputs_int = im_to_numpy(torch.clamp(inputs[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                imoutput_int = im_to_numpy(torch.clamp(imoutput[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                # imcoarse_int = im_to_numpy(torch.clamp(imcoarse[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                # imcoarsef_int = im_to_numpy(torch.clamp(imcoarsef[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                immask_int = immask[0][0].clamp(0.0, 1.0) * 255
                immask_int = immask_int.cpu().numpy().astype(np.uint8)
                # 储存图像,如果没有目录便创建,只保存_vx_前面那些字符
                # self.args.checkpoint.split("_vx")[0] 为存储位置
                # batches['name'][0]储存名
                # imfinal_int为储存结果
                dir = self.args.checkpoint
                if self.args.simage == 1:
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    view_dir = os.path.join(dir, "view")
                    if not os.path.exists(view_dir):
                        os.makedirs(view_dir)
                    from PIL import Image
                    view_image = Image.new('RGB', (1536, 256))
                    view_image.paste(Image.fromarray(inputs_int), (0, 0))
                    view_image.paste(Image.fromarray(target_int), (256, 0))
                    view_image.paste(Image.fromarray(imfinal_int), (512, 0))
                    view_image.paste(Image.fromarray(imoutput_int), (768, 0))
                    # view_image.paste(Image.fromarray(imcoarsef_int), (1024, 0))
                    # view_image.paste(Image.fromarray(imcoarse_int), (1280, 0))
                    view_image.paste(Image.fromarray(immask_int), (1024, 0))
                    view_image.paste(Image.fromarray(gtmask_int), (1280, 0))
                    view_image.save('%s/%s' % (view_dir, batches['name'][0]))
                # 计算转化为int后的图像的结果，如果测试的不是CLWD数据集
                psnr_2_o = compare_psnr(target_int,inputs_int)
                ssim_2_o = compare_ssim(target_int,inputs_int,multichannel=True)
                psnr_2 = compare_psnr(target_int,imfinal_int)
                ssim_2 = compare_ssim(target_int,imfinal_int,multichannel=True)
                target_int = torch.tensor(target_int)
                target_int = target_int.permute(2,0,1).unsqueeze(0).cuda()
                imfinal_int = torch.tensor(imfinal_int)
                imfinal_int = imfinal_int.permute(2,0,1).unsqueeze(0).cuda()
                inputs_int = torch.tensor(inputs_int)
                inputs_int = inputs_int.permute(2,0,1).unsqueeze(0).cuda()
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
        if self.args.is_realimg == 1:
            return
        end_time = time.time()
        total_time = end_time - start_time
        fps = total_time / 2025
        params = sum(p.numel() for p in self.model.parameters())/1000000.0
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')  # 得到当前时间字符串
        thismethod = self.args.mname  # 得到方法的名称
        trainning_dataset = self.args.trainset  # 得到训练集名称
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
        lpipses_2_avg_float = float(lpipses_2.avg.item())
        lpipses_2_o_avg_float = float(lpipses_2_o.avg.item())
        # 只放float类型的结果
        paper_data = {
            'time': current_time_str,
            'method': thismethod,
            'fps': round(fps,5),
            'params': round(params,3),
            'trainning dataset': trainning_dataset,
            'testing dataset': self.args.data,
            'PSNR': round(psnres_1.avg,3),
            'SSIM': round(ssimes_1_avg_float,4),
            'LPIPS': round(lpipses_1_avg_float,5),
            'f1': round(f1_meter.avg,5),
            'iou': round(iou_meter.avg,5),
            'PSNRO': round(psnres_1_o.avg,3),
            'SSIMO': round(ssimes_1_o_avg_float,4),
            'LPIPSO': round(lpipses_1_o_avg_float,5)
        }
        # paper_data = {
        #     'time': current_time_str,
        #     'method': thismethod,
        #     'fps': round(fps,5),
        #     'params': round(params,3),
        #     'trainning dataset': trainning_dataset,
        #     'testing dataset': self.args.data,
        #     'PSNR': round(psnres_2.avg,3),
        #     'SSIM': round(ssimes_2.avg,4),
        #     'LPIPS': round(lpipses_1_avg_float,5),
        #     'f1': round(f1_meter.avg,5),
        #     'iou': round(iou_meter.avg,5),
        #     'PSNRO': round(psnres_1_o.avg,3),
        #     'SSIMO': round(ssimes_2_o.avg,4),
        #     'LPIPSO': round(lpipses_1_o_avg_float,5)
        # }
        print(paper_data)
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
        floatfile = '/home/coolboy2/huangwenhong/SplitNet/float_results.csv'
        intfile = '/home/coolboy2/huangwenhong/SplitNet/int_results.csv'
        paperfile = '/home/coolboy2/huangwenhong/SplitNet/paper_res.csv'
        float_data_values = list(float_data.values())
        int_data_values = list(int_data.values())
        paper_data_values = list(paper_data.values())
        print("开始写入文件float_results.csv")
        with open(floatfile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(float_data_values)
        print("开始写入文件paper_res.csv")
        with open(paperfile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(paper_data_values)
        print("开始写入文件int_results.csv")
        with open(intfile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(int_data_values)
        print("写入完毕")
        print("DONE.\n")

   