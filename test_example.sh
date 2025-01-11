set -ex
K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res #'cat'
L1_LOSS=2
CONTENT_LOSS=2.5e-1
STYLE_LOSS=2.5e-1
PRIMARY_LOSS=0.01
IOU_LOSS=0.25
INPUT_SIZE=256
CUDA_VISIBLE_DEVICES=1 python test.py\
  -c result_imgs\            # the image saving dir
  --resume eval/watermark_removal_model_ckpts/MNet/10khigh/model_best.pth.tar\      # ckpt dir
  --arch  mnetold\           # model
  --machine testallmethods\  # machines to evaluate models
  --trainset 10khigh\        # dataset name
  --mname MNet\              # model name, only used when saving results to table
  --input-size 256\
  --test-batch 1\
  --is_realimg 0\
  --simage 1\
  --k1 5\
  --k2 2\
  --k3 1\
  --evaluate\
  --preprocess resize \
  --k_center ${K_CENTER} \
  --use_refine \
  --k_refine ${K_REFINE} \
  --k_skip_stage ${K_SKIP} \
  --base-dir watermark\   # dataset dir
  --data 10khigh
