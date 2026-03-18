#!/bin/bash
echo start

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12390" \
    ./train.py \
    --config configs/SSP_SAM_CLIP_B_FT_unc.py \
    --clip_pretrained 'pretrained_checkpoints/CS/CS-ViT-B-16.pt' \

# -----------------------------
# Examples (uncomment to use)
# -----------------------------

# train grefcoco (L/336)
# torchrun --nproc_per_node="8" \
#     --nnodes="1" \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12390" \
#     ./train.py \
#     --config configs/SSP_SAM_CLIP_L_FT_grefcoco.py \
#     --clip_pretrained 'pretrained_checkpoints/CS/CS-ViT-L-14-336px.pt'
