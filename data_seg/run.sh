#!/usr/bin/env bash

# refcoco (unc)
python ./data_seg/data_process.py --data_root ./data_seg --output_dir ./data_seg --dataset refcoco --split unc --generate_mask

# refcoco+ (unc)
# python ./data_seg/data_process.py --data_root ./data_seg --output_dir ./data_seg --dataset refcoco+ --split unc --generate_mask

# refcocog (umd)
# python ./data_seg/data_process.py --data_root ./data_seg --output_dir ./data_seg --dataset refcocog --split umd --generate_mask

# refcocog (google)
# python ./data_seg/data_process.py --data_root ./data_seg --output_dir ./data_seg --dataset refcocog --split google --generate_mask

# refclef (unc)
# python ./data_seg/data_process.py --data_root ./data_seg --output_dir ./data_seg --dataset refclef --split unc --generate_mask

# refclef (berkeley)
# python ./data_seg/data_process.py --data_root ./data_seg --output_dir ./data_seg --dataset refclef --split berkeley --generate_mask

# grefcoco:
# 当前 grefcoco_data_process.py 内有硬编码 data_root/output_dir，请先改脚本再运行。
# python ./data_seg/grefcoco_data_process.py
