#!/bin/bash
echo start

# test refcoco
torchrun --nproc_per_node=1 --master_port=29590 test.py --config configs/SSP_SAM_CLIP_L_FT_unc.py --test_split testB --clip_pretrained 'pretrained_checkpoints/CS/CS-ViT-L-14-336px.pt' --checkpoint output/reimplement_unc/checkpoint_best_miou.pth

# -----------------------------
# Examples (uncomment to use)
# -----------------------------

# test refcoco+ (testB)
# torchrun --nproc_per_node=1 --master_port=29590 test.py \
#   --config configs/SSP_SAM_CLIP_B_FT_unc+.py \
#   --test_split testB \
#   --clip_pretrained 'pretrained_checkpoints/CS/CS-ViT-B-16.pt' \
#   --checkpoint output/your_save_folder/checkpoint_best_miou.pth
