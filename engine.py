import math
import os
import sys
from typing import Iterable
import torch
from statistics import mean, stdev
import util.misc as utils
from util import box_ops

import logging
import torch.distributed as dist
import time
import datetime
from tqdm import tqdm
from metric.mask_op import mask_iou, mask_pair_iou

class data_prefetcher():
    def __init__(self, loader, device):
        self.length = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            samples, targets, image_sam, img_mask, task_id = next(self.loader)
            self.next_img, self.next_mask = samples.decompose()
            self.next_target = targets
            self.next_image_sam = image_sam
            self.next_img_mask = img_mask
            self.next_task_id = task_id
        except StopIteration:
            self.next_img = self.next_mask = self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.to(self.device, non_blocking=True)
            self.next_mask = self.next_mask.to(self.device, non_blocking=True)
            tensor_dict = self.next_target.tensor_dict
            self.next_target.tensor_dict = {k: tensor_dict[k].to(self.device, non_blocking=True) for k in tensor_dict}
            self.next_image_sam = self.next_image_sam.to(self.device, non_blocking=True)
            self.next_img_mask = self.next_img_mask.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img, mask, target, image_sam, img_mask, task_id = self.next_img, self.next_mask, self.next_target, self.next_image_sam, self.next_img_mask, self.next_task_id
        self.preload()
        return img, mask, target, image_sam, img_mask, task_id

    def __next__(self):
        img, mask, target, image_sam, img_mask, task_id = self.next()
        if img == None:
            raise StopIteration
        return img, mask, target, image_sam, img_mask, task_id

    def __iter__(self):
        return self

    def __len__(self):
        return self.length


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    model.train()
    criterion_res = criterion['res']
    criterion_rec = criterion['rec']
    criterion_res.train()
    criterion_rec.train()

    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'

    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)
    img, mask, target, image_sam, img_mask, task_id = prefetcher.next()
    iteration = 0
    while img is not None:
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        ori_size = target_dict['ori_size']
        iteration = iteration + 1
        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask, image_sam, ori_size, target_dict)
        pred_masks = outputs.get('pred_masks')
        pred_boxes = outputs.get('pred_boxes')

        if pred_masks is not None:
            # loss for res
            loss_dict = criterion_res(pred_masks, img_mask)
            weight_dict = criterion_res.weight_dict
            losses_res = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes (res)
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()

            # loss for rec
            loss_dict_rec = criterion_rec(outputs, target_dict)
            weight_dict_rec = criterion_rec.weight_dict
            losses_rec = sum(loss_dict_rec[k] * weight_dict_rec[k] for k in loss_dict_rec.keys() if k in weight_dict_rec)

            # reduce losses over all GPUs for logging purposes (rec)
            loss_dict_reduced_rec = utils.reduce_dict(loss_dict_rec)
            loss_dict_reduced_scaled_rec = {k: v * weight_dict_rec[k]
                                        for k, v in loss_dict_reduced_rec.items() if k in weight_dict_rec}
            losses_reduced_scaled_rec = sum(loss_dict_reduced_scaled_rec.values())
            loss_value_rec = losses_reduced_scaled_rec.item()

            losses = losses_res + losses_rec


            if not math.isfinite(loss_value) or not math.isfinite(loss_value_rec):
                print("Loss is {}, stopping training".format(loss_value))
                print("Loss is {}, stopping training".format(loss_value_rec))
                print(loss_dict_reduced)
                print(loss_dict_reduced_rec)
                sys.exit(1)

            optimizer.zero_grad()
            
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            iter_time.update(time.time() - end)
            end = time.time()
            metric_logger.update(loss=loss_value+loss_value_rec, **loss_dict_reduced_scaled, **loss_dict_reduced_scaled_rec)

            if iteration % 100 == 0 or iteration == max_iter:
                eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs-epoch-1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    metric_logger.delimiter.join(
                        [header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}"
                        ]
                    ).format(
                        epoch=epoch+1, iter=iteration, max_iter=max_iter,
                        lr=optimizer.param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))
            img, mask, target, image_sam, img_mask, task_id = prefetcher.next()

        else:
                # 说明当前在pretrain
                # loss for rec
                loss_dict_rec = criterion_rec(outputs, target_dict)
                weight_dict_rec = criterion_rec.weight_dict
                losses_rec = sum(loss_dict_rec[k] * weight_dict_rec[k] for k in loss_dict_rec.keys() if k in weight_dict_rec)

                # reduce losses over all GPUs for logging purposes (rec)
                loss_dict_reduced_rec = utils.reduce_dict(loss_dict_rec)
                loss_dict_reduced_scaled_rec = {k: v * weight_dict_rec[k]
                                            for k, v in loss_dict_reduced_rec.items() if k in weight_dict_rec}
                losses_reduced_scaled_rec = sum(loss_dict_reduced_scaled_rec.values())
                loss_value_rec = losses_reduced_scaled_rec.item()

                losses = losses_rec


                if not math.isfinite(loss_value_rec):
                    print("Loss is {}, stopping training".format(loss_value_rec))
                    print(loss_dict_reduced_rec)
                    sys.exit(1)

                optimizer.zero_grad()
                
                losses.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

                iter_time.update(time.time() - end)
                end = time.time()
                metric_logger.update(loss=loss_value_rec, **loss_dict_reduced_scaled_rec)

                if iteration % 100 == 0 or iteration == max_iter:
                    eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs-epoch-1))
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    logger.info(
                        metric_logger.delimiter.join(
                            [header,
                            "lr: {lr}",
                            "eta: {eta}",
                            "time: {time}",
                            "data: {data}",
                            "memory: {memory:.0f}",
                            "{meters}"
                            ]
                        ).format(
                            epoch=epoch+1, iter=iteration, max_iter=max_iter,
                            lr=optimizer.param_groups[0]["lr"],
                            eta=eta_string,
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                            meters=str(metric_logger)
                        ))
                img, mask, target, image_sam, img_mask, task_id = prefetcher.next()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessor, data_loader, device, save_path=''):
    model.eval()

    criterion_res = criterion['res']
    criterion_rec = criterion['rec']
    if criterion_res:
        criterion_res.eval()
    if criterion_rec:
        criterion_rec.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    
    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    accum_acc_mask = 0
    accum_intersections = 0
    accum_unions = 0
    accum_acc_bbox = 0
    accum_iou_bbox = 0
    accum_sample_bbox = 0

    # 对GRES的拓展，增加对gt为空的样本的统计
    num_empty_gt = 0
    num_empty_pred = 0
    num_not_empty_gt = 0
    num_not_empty_pred = 0
    # 
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher = data_prefetcher(data_loader, device)
    multi_task_flag = False

    times = []
    for iteration, (img, mask, target, image_sam, img_mask, task_id) in enumerate(tqdm(prefetcher)):

        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        
        ori_size = target_dict['ori_size']
        gt_bbox = target_dict['orig_bbox']

        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask, image_sam, ori_size, target_dict)
        pred_masks = outputs.get('pred_masks')
        pred_boxes = outputs.get('pred_boxes')

        ####################################################
        # save
        # # multi_modal feature visualization
        # multi_modal_feat = outputs['x_multi_modal']
        # # np.save(f"visualization/unc/multi_modal_feat_embedding/testB/{str(iteration)}.npy", multi_modal_feat)
        # fused_img_feat = outputs['fuse_img_feat']
        # verify_score = outputs['verify_score']
        # multi_prompt = outputs['multi_prompt']
        # sam_upscaled_embedding = outputs['upscaled_embedding']
        # np.save(f"visualization/gref/sam_upscaled_embedding/test/{str(iteration)}.npy", sam_upscaled_embedding)
        ####################################################

        if pred_masks is not None:
            multi_task_flag = True

        if multi_task_flag:
            if criterion_res and criterion_rec:
                # for res
                loss_dict = criterion_res(pred_masks, img_mask)
                weight_dict = criterion_res.weight_dict

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                loss_value = sum(loss_dict_reduced_scaled.values()).item()

                # for rec
                loss_dict_rec = criterion_rec(outputs, target_dict)
                weight_dict_rec = criterion_rec.weight_dict

                loss_dict_reduced_rec = utils.reduce_dict(loss_dict_rec)
                loss_dict_reduced_scaled_rec = {k: v * weight_dict_rec[k]
                                            for k, v in loss_dict_reduced_rec.items() if k in weight_dict_rec}
                loss_value_rec = sum(loss_dict_reduced_scaled_rec.values()).item()
                

                metric_logger.update(loss=loss_value+loss_value_rec, **loss_dict_reduced_scaled, **loss_dict_reduced_scaled_rec)
        else:
            if criterion_rec:
                # for rec
                loss_dict_rec = criterion_rec(outputs, target_dict)
                weight_dict_rec = criterion_rec.weight_dict

                loss_dict_reduced_rec = utils.reduce_dict(loss_dict_rec)
                loss_dict_reduced_scaled_rec = {k: v * weight_dict_rec[k]
                                            for k, v in loss_dict_reduced_rec.items() if k in weight_dict_rec}
                loss_value_rec = sum(loss_dict_reduced_scaled_rec.values()).item()
                

                metric_logger.update(loss=loss_value_rec, **loss_dict_reduced_scaled_rec)
        
        if multi_task_flag:
            # ##########################################################33
            # written by wayne
            # mod bbox_iou to mask_iou
            # for res
            pred_mask = (torch.sigmoid(outputs['pred_masks']) >= 0.5).detach()
            
            gt_mask = img_mask
            
            ious, intersections, unions = mask_pair_iou(pred_mask, gt_mask, target_dict=target_dict)

            # statis_mask_ious.append(round(float(ious.detach().cpu()), 4))
            
            # 对GRES的拓展，增加对空目标的处理
            gt_area =gt_mask.flatten(1).sum(dim=1)
            if gt_area.sum() == 0:
                num_empty_gt += 1
            else:
                num_not_empty_gt += 1
            
            if ious.sum() == 1:
                num_empty_pred += 1
            elif ious.sum() > 0 and ious.sum() < 1:
                num_not_empty_pred += 1
            elif ious.sum() == 0:
                ...
            num_acc_mask = (ious[:, None] >= iou_thrs[None]).sum(dim=0)
            accum_acc_mask += num_acc_mask
            # 
            sum_intersections = intersections.sum()
            sum_unions = unions.sum()
            sum_iou = ious.sum()
            num_acc = (ious[:, None] >= iou_thrs[None]).sum(dim=0)
            # num_acc = sum_iou
            num_sample = torch.as_tensor(img.size(0), device=img.device)

            # for rec
            # 
            pred_boxes = postprocessor(outputs, target_dict)
            ious_bbox = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
            sum_iou_bbox = ious_bbox.sum()
            num_acc_bbox = (ious_bbox[:, None] >= iou_thrs[None]).sum(dim=0)
            num_sample_bbox = torch.as_tensor(img.size(0), device=img.device)
            accum_acc_bbox += num_acc_bbox
            accum_iou_bbox += sum_iou_bbox
            accum_sample_bbox += num_sample_bbox

            accum_acc += num_acc
            accum_iou += sum_iou
            accum_sample += num_sample
            accum_intersections += sum_intersections
            accum_unions += sum_unions

            iter_time.update(time.time() - end)
            end = time.time()
        else:
            # ##########################################################33
            # written by wayne
            # mod bbox_iou to mask_iou
            # for rec
            pred_boxes = postprocessor(outputs, target_dict)
            ious_bbox = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
            sum_iou_bbox = ious_bbox.sum()
            num_acc_bbox = (ious_bbox[:, None] >= iou_thrs[None]).sum(dim=0)
            num_sample_bbox = torch.as_tensor(img.size(0), device=img.device)
            accum_acc_bbox += num_acc_bbox
            accum_iou_bbox += sum_iou_bbox
            accum_sample_bbox += num_sample_bbox

            iter_time.update(time.time() - end)
            end = time.time()

    # accumulate predictions from all images
    # print(statis_mask_ious)
    if multi_task_flag:
        if utils.get_world_size() > 1:
            # for res
            dist.all_reduce(accum_acc)
            dist.all_reduce(accum_iou)
            dist.all_reduce(accum_sample)
            dist.all_reduce(accum_intersections)
            dist.all_reduce(accum_unions)
            
            dist.all_reduce(num_empty_gt)
            dist.all_reduce(num_empty_pred)
            dist.all_reduce(num_not_empty_gt)
            dist.all_reduce(num_not_empty_pred)


            # for rec
            dist.all_reduce(accum_acc_bbox)
            dist.all_reduce(accum_iou_bbox)
            dist.all_reduce(accum_sample_bbox)

        # for res
        # acc = accum_acc / accum_sample.float().item()
        # for rec
        acc = accum_acc_bbox / accum_sample_bbox.float().item()
        mask_acc = accum_acc_mask / accum_sample.float().item()
        giou = accum_iou.item() / accum_sample.float().item()
        ciou = accum_intersections.item() / accum_unions.item()
        
        val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
        val_acc.update({'G_iou(Mean_iou)': giou})
        val_acc.update({'C_iou(Overal_iou)': ciou})
        val_acc.update({f'Mask_Pr@{t:.2f}': a.item() for t, a in zip(iou_thrs, mask_acc)})

        if num_empty_gt != 0:
            no_target_acc = num_empty_pred / num_empty_gt
            target_acc = num_not_empty_pred / num_not_empty_gt
            val_acc.update({'No_target_acc': no_target_acc})
            val_acc.update({'target_acc': target_acc})
        
        val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}
    else:
        if utils.get_world_size() > 1:
            # for rec
            dist.all_reduce(accum_acc_bbox)
            dist.all_reduce(accum_iou_bbox)
            dist.all_reduce(accum_sample_bbox)

        # for res
        # acc = accum_acc / accum_sample.float().item()
        # for rec
        acc = accum_acc_bbox / accum_sample_bbox.float().item()
        miou = accum_iou_bbox.item() / accum_sample_bbox.float().item()

        val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
        val_acc.update({'Mean_iou': miou})
        val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}
    return val_stats, val_acc, val_time



@torch.no_grad()
def evaluate_phrase_cut(model, criterion, postprocessor, data_loader, device, save_path=''):
    model.eval()

    criterion_res = criterion['res']
    criterion_rec = criterion['rec']
    if criterion_res:
        criterion_res.eval()
    if criterion_rec:
        criterion_rec.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    accum_intersections = 0
    accum_unions = 0
    accum_acc_bbox = 0
    accum_iou_bbox = 0
    accum_sample_bbox = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher = data_prefetcher(data_loader, device)
    multi_task_flag = False
    for iteration, (img, mask, target, image_sam, img_mask, task_id) in enumerate(tqdm(prefetcher)):
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        
        ori_size = target_dict['ori_size']
        gt_bbox = target_dict['orig_bbox']

        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask, image_sam, ori_size, target_dict)
        pred_masks = outputs.get('pred_masks')
        pred_boxes = outputs.get('pred_boxes')

        if pred_masks is not None:
            multi_task_flag = True

        if multi_task_flag:
            if criterion_res and criterion_rec:
                # for res
                loss_dict = criterion_res(pred_masks, img_mask)
                weight_dict = criterion_res.weight_dict

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                loss_value = sum(loss_dict_reduced_scaled.values()).item()

                # for rec
                loss_dict_rec = criterion_rec(outputs, target_dict)
                weight_dict_rec = criterion_rec.weight_dict

                loss_dict_reduced_rec = utils.reduce_dict(loss_dict_rec)
                loss_dict_reduced_scaled_rec = {k: v * weight_dict_rec[k]
                                            for k, v in loss_dict_reduced_rec.items() if k in weight_dict_rec}
                loss_value_rec = sum(loss_dict_reduced_scaled_rec.values()).item()
                

                metric_logger.update(loss=loss_value+loss_value_rec, **loss_dict_reduced_scaled, **loss_dict_reduced_scaled_rec)
        else:
            if criterion_rec:
                # for rec
                loss_dict_rec = criterion_rec(outputs, target_dict)
                weight_dict_rec = criterion_rec.weight_dict

                loss_dict_reduced_rec = utils.reduce_dict(loss_dict_rec)
                loss_dict_reduced_scaled_rec = {k: v * weight_dict_rec[k]
                                            for k, v in loss_dict_reduced_rec.items() if k in weight_dict_rec}
                loss_value_rec = sum(loss_dict_reduced_scaled_rec.values()).item()
                

                metric_logger.update(loss=loss_value_rec, **loss_dict_reduced_scaled_rec)
        
        if multi_task_flag:
            # ##########################################################33
            # written by wayne
            # mod bbox_iou to mask_iou
            # for res
            pred_mask = (torch.sigmoid(outputs['pred_masks']) >= 0.5).detach()
            gt_mask = img_mask
            ious, intersections, unions = mask_pair_iou(pred_mask, gt_mask, target_dict=target_dict)
            sum_intersections = intersections.sum()
            sum_unions = unions.sum()
            sum_iou = ious.sum()
            num_acc = (ious[:, None] >= iou_thrs[None]).sum(dim=0)
            # num_acc = sum_iou
            num_sample = torch.as_tensor(img.size(0), device=img.device)

            # for rec
            # 
            # sam_pred_bboxes = mask2box(pred_mask, ori_size)
            pred_boxes = postprocessor(outputs, target_dict)
            ious_bbox = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
            sum_iou_bbox = ious_bbox.sum()
            num_acc_bbox = (ious_bbox[:, None] >= iou_thrs[None]).sum(dim=0)
            num_sample_bbox = torch.as_tensor(img.size(0), device=img.device)
            accum_acc_bbox += num_acc_bbox
            accum_iou_bbox += sum_iou_bbox
            accum_sample_bbox += num_sample_bbox

            accum_acc += num_acc
            accum_iou += sum_iou
            accum_sample += num_sample
            accum_intersections += sum_intersections
            accum_unions += sum_unions

            iter_time.update(time.time() - end)
            end = time.time()
        else:
            # ##########################################################33
            # written by wayne
            # mod bbox_iou to mask_iou
            # for rec
            pred_boxes = postprocessor(outputs, target_dict)
            ious_bbox = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
            sum_iou_bbox = ious_bbox.sum()
            num_acc_bbox = (ious_bbox[:, None] >= iou_thrs[None]).sum(dim=0)
            num_sample_bbox = torch.as_tensor(img.size(0), device=img.device)
            accum_acc_bbox += num_acc_bbox
            accum_iou_bbox += sum_iou_bbox
            accum_sample_bbox += num_sample_bbox

            iter_time.update(time.time() - end)
            end = time.time()           

    # accumulate predictions from all images
    if multi_task_flag:
        if utils.get_world_size() > 1:
            # for res
            dist.all_reduce(accum_acc)
            dist.all_reduce(accum_iou)
            dist.all_reduce(accum_sample)
            dist.all_reduce(accum_intersections)
            dist.all_reduce(accum_unions)


            # for rec
            dist.all_reduce(accum_acc_bbox)
            dist.all_reduce(accum_iou_bbox)
            dist.all_reduce(accum_sample_bbox)

        # for res
        # acc = accum_acc / accum_sample.float().item()
        # for rec
        acc = accum_acc_bbox / accum_sample_bbox.float().item()
        miou = accum_iou.item() / accum_sample.float().item()
        oiou = accum_intersections.item() / accum_unions.item()

        val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
        val_acc.update({'G_iou(Mean_iou)': miou})
        val_acc.update({'C_iou(Overal_iou)': oiou})
        val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}
    else:
        if utils.get_world_size() > 1:
            # for rec
            dist.all_reduce(accum_acc_bbox)
            dist.all_reduce(accum_iou_bbox)
            dist.all_reduce(accum_sample_bbox)

        # for res
        # acc = accum_acc / accum_sample.float().item()
        # for rec
        acc = accum_acc_bbox / accum_sample_bbox.float().item()
        miou = accum_iou_bbox.item() / accum_sample_bbox.float().item()

        val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
        val_acc.update({'Mean_iou': miou})
        val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}
    return val_stats, val_acc, val_time
