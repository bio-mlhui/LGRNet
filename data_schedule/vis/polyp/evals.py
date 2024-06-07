from data_schedule.vis.evaluator_utils import register_vis_metric
import os
import torch
import detectron2.utils.comm as comm
import logging
import subprocess


@register_vis_metric
def polyp_metric_aggregator(metrics_by_vid_frame, dataset_meta, eval_meta_keys, **kwargs):
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
    for billie_eilish in eval_meta_keys:
        mean_iou_by_each_video[billie_eilish] = torch.tensor([metrics_by_vid_frame[billie_eilish][fname]['iou'] for fname in eval_meta_keys[billie_eilish]]).mean()
        mean_dice_by_each_video[billie_eilish] = torch.tensor([metrics_by_vid_frame[billie_eilish][fname]['dice'] for fname in eval_meta_keys[billie_eilish]]).mean()
        
    mean_iou_by_each_video = dict(sorted(mean_iou_by_each_video.items(), key=lambda x: x[1]))
    mean_dice_by_each_video = dict(sorted(mean_dice_by_each_video.items(), key=lambda x: x[1]))    
    logging.debug(f'mean_iou_by_each_video: {mean_iou_by_each_video}')
    logging.debug(f'mean_dice_by_each_video: {mean_dice_by_each_video}')
    
    return eval_metrics


