# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from signal import pthread_sigmask
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def mask_iou(mask1, mask2):
    """
    :param mask1:  l
    :param mask2:  l
    :return: iou
    """
    mask1 =mask1.reshape([-1])
    mask2=mask2.reshape([-1])
    t = np.array(mask1 > 0.5)
    p = mask2 > 0.
    intersection = np.logical_and(t, p)
    union = np.logical_or(t, p)
    iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)

    ap = dict()
    thresholds = np.arange(0.5, 1, 0.05)
    s = []
    for thresh in thresholds:
        ap[thresh] = float(iou > thresh)
    return iou,ap


def mask_processing(mask,info_img):
    # print(info_img)
    h, w, nh, nw, dx, dy,_=info_img
    # print(info_img)
    # print(mask)
    mask=mask[dy:dy + nh, dx:dx + nw,None]
    mask=cv2.resize(mask,(int(w),int(h)))
    return mask


def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union


def binaryMaskIOU(mask1, mask2):
    mask1_area = torch.count_nonzero(mask1)
    mask2_area = torch.count_nonzero(mask2)
    intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou

# 提交给TIP的代码，没有考虑空目标的情况，这个在常规数据集是不会报错的
# def mask_iou_reftr(masks, target):
#     assert(target.shape[-2:] == masks.shape[-2:])
#     I = torch.sum(torch.logical_and(masks, target))
#     U = torch.sum(torch.logical_or(masks, target))
#     return I.float() / U.float(), I, U

# 修改版提交TCSVT的代码，针对grefcoco考虑了空目标
def mask_iou_reftr(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = torch.sum(torch.logical_and(masks, target))
    U = torch.sum(torch.logical_or(masks, target))
    
    # 检查 gt_mask 是否全为 0
    if torch.all(target == 0):
        # 如果 pred_mask 也全为 0，返回 IoU 为 1
        if torch.all(masks == 0):
            return torch.tensor(1.0, device=masks.device), torch.tensor(0, dtype=torch.long, device=masks.device), torch.tensor(0, dtype=torch.long, device=masks.device)
        # 如果 pred_mask 不全为 0，返回 IoU 为 0
        elif torch.count_nonzero(masks) < 50:
            return torch.tensor(1.0, device=masks.device), torch.tensor(0, dtype=torch.long, device=masks.device), torch.tensor(0, dtype=torch.long, device=masks.device)
        else:
            return torch.tensor(0.0, device=masks.device), torch.tensor(0, dtype=torch.long, device=masks.device), U
    else:
        # 避免除以零
        if U == 0:
            return torch.tensor(0.0, device=masks.device), torch.tensor(0, dtype=torch.long, device=masks.device), torch.tensor(0, dtype=torch.long, device=masks.device)
        else:
            return I.float() / U.float(), I, U


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)




def mask_pair_iou(pred_mask, gt_mask, target_dict):
    assert pred_mask.shape == gt_mask.shape, "pred_mask should have the same shape with gt_mask"
    bs = pred_mask.shape[0]

    ori_sizes = target_dict['ori_size']
    # ious = [mask_iou_reftr(pred_mask[i], gt_mask[i]) for i in range(0, bs)]
    ious = []
    Intersections = []
    Unions = []
    
    for i in range(0, bs):
        # pred_mask_ori = F.interpolate(pred_mask[i][None].float(), size = (ori_sizes[i][0], ori_sizes[i][1]), mode='nearest').bool()
        # gt_mask_ori = F.interpolate(gt_mask[i][None], size = (ori_sizes[i][0], ori_sizes[i][1]), mode='nearest').bool()
        # iou, I, U = mask_iou_reftr(pred_mask_ori, gt_mask_ori)
        iou, I, U = mask_iou_reftr(pred_mask[i], gt_mask[i])
        ious.append(iou)
        Intersections.append(I)
        Unions.append(U)
    ious_tensor = torch.stack(ious)
    Intersections_tensor = torch.stack(Intersections)
    Unions_tensor = torch.stack(Unions)
    return ious_tensor, Intersections_tensor, Unions_tensor