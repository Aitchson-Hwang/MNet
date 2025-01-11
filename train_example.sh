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
DATASET=10kgray
NAME=sunet

CUDA_VISIBLE_DEVICES=1 python main.py  
 --epochs 200\
 --schedule 200 \
 --lr 1e-3 \
 -c eval\     		# ckpt saving dir
 --arch mnetold\    # model to train
 --k1 5\			# number of unet
 --k2 2\
 --k3 1\
 --sltype vggx\
 --inlr 2e-4\
 --fnlr 1e-6\
 --mask_mode ${MASK_MODE}\
 --is_clip 1\
 --lambda_content ${CONTENT_LOSS} \
 --lambda_style ${STYLE_LOSS} \
 --lambda_l1 ${L1_LOSS} \
 --masked True\
 --loss-type hybrid\
 --limited-dataset 1\
 --machine sunet\
 --input-size ${INPUT_SIZE} \
 --crop_size ${INPUT_SIZE} \
 --train-batch 16 \
 --test-batch 1 \
 --preprocess resize \
 --k_center ${K_CENTER} \
 --use_refine \
 --k_refine ${K_REFINE} \
 --k_skip_stage ${K_SKIP} \
 --base-dir watermark\		# dataset dir
 --data 10khigh				# dataset name

