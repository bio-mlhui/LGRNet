from data_schedule.vis.evaluator_utils import register_vis_metric
import os
from glob import glob
from tqdm import tqdm
import shutil
from functools import partial
from PIL import Image
import numpy as np
import torch
import detectron2.utils.comm as comm
import logging
import pycocotools.mask as mask_util
from pycocotools.mask import decode as decode_rle
import data_schedule.vis.fibroid.metrics as metrics

@register_vis_metric
def fibroid_other_medi(model_preds, 
                     dataset_meta,
                     **kwargs):
    assert comm.is_main_process()

    iou_by_test_sample = []
    dice_by_test_sample = []
    preds_by_test_sample = []
    gt_by_test_sample = []
    get_frames_gt_mask_fn = dataset_meta.get('get_frames_gt_mask_fn')

    for pred in model_preds:
        video_id = pred['video_id'] # str
        frame_name = pred['frame_name'] # list[str], t'
        masks = pred['masks']# list[rle], nq
        scores = pred['scores'] # nq

        max_idx = torch.tensor(scores).argmax()
        pred_mask = masks[max_idx] # rle
        pred_mask = decode_rle(pred_mask)
        pred_mask = torch.as_tensor(pred_mask, dtype=torch.uint8).contiguous() # h w

        gt_mask, _ = get_frames_gt_mask_fn(video_id=video_id, frames=[frame_name]) # 1 h w
        gt_mask = gt_mask[0].int() # 0/1

        preds_by_test_sample.append(pred_mask)
        gt_by_test_sample.append(gt_mask)

        tp, fp, fn, tn = metrics.get_stats(pred_mask[None, None, ...], gt_mask[None, None, ...], 
                                           mode='binary')
        iou_score = metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        dice = metrics.dice(tp, fp, fn, tn, reduction='micro')

        iou_by_test_sample.append(iou_score)
        dice_by_test_sample.append(dice)

    mean_iou = torch.tensor(iou_by_test_sample).mean()
    mean_dice = torch.tensor(dice_by_test_sample).mean()

    preds_by_test_sample = torch.stack(preds_by_test_sample, dim=0).unsqueeze(1) # N 1 h w
    gt_by_test_sample = torch.stack(gt_by_test_sample, dim=0).unsqueeze(1) # N 1 h w

    tp, fp, fn, tn = metrics.get_stats(preds_by_test_sample, gt_by_test_sample, 
                                        mode='binary')
    overall_iou = metrics.iou_score(tp, fp, fn, tn, reduction='micro')    
    recall = metrics.recall(tp, fp, fn, tn, reduction='micro-imagewise') 
    precision = metrics.precision(tp, fp, fn, tn, reduction='micro-imagewise')

    all_medi = {
        'mean_iou': mean_iou,
        'dice': mean_dice,
        'overall_iou': overall_iou, # J/overallIoU
        'recall': recall,
        'precision': precision,
        'F': 2 * precision * recall / (precision + recall)
    }  
    return all_medi   
    

from collections import defaultdict

# by_vid, by_frame
iou_dict = defaultdict(dict)

@register_vis_metric
def fibroid_mask_dice_iou(frame_pred, dataset_meta, **kwargs):
    video_id = frame_pred['video_id']
    frame_name = frame_pred['frame_name']
    masks = frame_pred['masks'] # nq h w
    get_frames_gt_mask_fn = dataset_meta.get('get_frames_gt_mask_fn')
    scores = torch.tensor(frame_pred['classes']) # nq c, 保证c是2
    foreground_scores = scores[:, :-1].sum(-1) # nq
    max_idx = foreground_scores.argmax()
    pred_mask = masks[max_idx].int() # h w

    gt_mask, _ = get_frames_gt_mask_fn(video_id=video_id, frames=[frame_name]) # 1 h w
    gt_mask = gt_mask[0].int() # h w

    inter, union    = (pred_mask*gt_mask).sum(), (pred_mask+gt_mask).sum()
    dice = (2*inter+1)/(union+1)
    iou = (inter+1)/(union-inter+1)
    iou_dict[video_id][frame_name] = iou
    if iou > 0.6:
        print(f'video_id: {video_id}, frame: {frame_name}: dice {dice}, iou {iou}')
    return {'dice': dice, 'iou': iou}

@register_vis_metric
def fibroid_metric_aggregator(metrics_by_vid_frame, dataset_meta, eval_meta_keys, **kwargs):
    # output: eval_metrics
    # video: frame_name: metric/ vid_metrics

    eval_metrics = {}
    # video, frame_name
    # perframe metrics
    metric_names = metrics_by_vid_frame[list(eval_meta_keys.keys())[0]][eval_meta_keys[list(eval_meta_keys.keys())[0]][0]]
    for taylor_swift in metric_names:
        eval_metrics[taylor_swift] = torch.tensor([metrics_by_vid_frame[video][frame][taylor_swift]  for video in eval_meta_keys.keys() for frame in eval_meta_keys[video]]).mean()
    
    # metrics by each video
    mean_iou_by_each_video = {}
    mean_dice_by_each_video = {}
    for video in eval_meta_keys:
        mean_iou_by_each_video[video] = torch.tensor([metrics_by_vid_frame[video][fname]['iou'] for fname in eval_meta_keys[video]]).mean()
        mean_dice_by_each_video[video] = torch.tensor([metrics_by_vid_frame[video][fname]['dice'] for fname in eval_meta_keys[video]]).mean()
    
    mean_iou_by_each_video = dict(sorted(mean_iou_by_each_video.items(), key=lambda x: x[1]))
    mean_dice_by_each_video = dict(sorted(mean_dice_by_each_video.items(), key=lambda x: x[1]))
    logging.debug(f'mean_iou_by_each_video: {mean_iou_by_each_video}')
    logging.debug(f'mean_dice_by_each_video: {mean_dice_by_each_video}')
    return eval_metrics

