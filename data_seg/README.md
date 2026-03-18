# data_seg 数据预处理说明

本目录用于将各类 referring segmentation 数据集转换为项目训练/测试使用的统一标注格式，并按需生成 mask。

## 目录与脚本

- `data_process.py`: 处理 `refcoco` / `refcoco+` / `refcocog` / `refclef`
- `grefcoco_data_process.py`: 处理 `grefcoco`（脚本内当前有硬编码路径，需先改）
- `anns/`: 预处理后的标注 json 输出目录
- `masks/`: 预处理后的 mask `.npy` 输出目录（开启 `--generate_mask` 时生成）

## 输出格式

主输出文件为 `data_seg/anns/<dataset>.json`，按 split 组织，样例字段：

- `iid`: 图像 id
- `bbox`: 框坐标
- `cat_id`: 类别 id
- `refs`: 文本描述（train 为多句，val/test 按句展开）
- `mask_id`: mask 对应 id

mask 文件输出到 `data_seg/masks/<dataset>/<mask_id>.npy`。

## 快速使用

在项目根目录执行（以 `refcoco` 为例）：

```bash
python ./data_seg/data_process.py \
  --data_root ./data_seg \
  --output_dir ./data_seg \
  --dataset refcoco \
  --split unc \
  --generate_mask
```

更多示例见 `data_seg/run.sh`。

## 参数说明（data_process.py）

- `--data_root`: 数据根目录（应包含对应数据集目录与图像）
- `--output_dir`: 输出目录（通常使用 `./data_seg`）
- `--dataset`: `refcoco | refcoco+ | refcocog | refclef`
- `--split`: 数据划分来源（常见：`unc`、`umd`、`google`）
- `--generate_mask`: 是否额外生成 `.npy` mask

## grefcoco 说明

`grefcoco_data_process.py` 目前在脚本底部写死了 `data_root` 路径和部分参数，不是通用 argparse 入口。使用前请先按本地路径修改脚本中的：

- `data_root`
- `dataset`（保持 `grefcoco`）
- `split`（通常 `unc`）
- 输出目录参数
