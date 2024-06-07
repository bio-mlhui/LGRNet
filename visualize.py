from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.visualizer import ColorMode
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
import networkx as nx

class MyVisualizer(Visualizer):
    def _jitter(self, color):
        return color

def generate_instance_canvas(vid_frames, metadata, H, W, pred_mask):
    """pred_mask: h w, score:float"""
    istce_canvas = MyVisualizer(img_rgb=vid_frames*255, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
    istce = Instances([H, W], 
        pred_masks=pred_mask.unsqueeze(0), # 1 H W
        scores=torch.tensor([1]), # 1,
        pred_classes=torch.tensor([0]) # 1,
    )
    istce_canvas.draw_instance_predictions(istce)
    istce_canvas = istce_canvas.get_output()
    return istce_canvas.get_image()

def generate_instance_canvas_uou(vid_frames, metadata, H, W, pred_mask, color_dataset):
    """pred_mask: h w, score:float"""
    from detectron2.data import MetadataCatalog
    metadata = MetadataCatalog.get(color_dataset)
    istce_canvas = MyVisualizer(img_rgb=vid_frames, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
    
    istce = Instances([H, W], 
        pred_masks=pred_mask.unsqueeze(0), # 1 H W
        scores=torch.tensor([1]), # 1,
        pred_classes=torch.tensor([0]) # 1,
    )
    istce_canvas.draw_instance_predictions(istce)
    istce_canvas = istce_canvas.get_output()
    return istce_canvas.get_image()

from torchvision.io import write_video
import random
def save_model_output(videos, directory, file_name, pred_masks=None, scores=None, color=None, fps=None):
    # t 3 h w
    # nq t h w
    # vi nq 
    # t h w
    vid_frames = videos.detach().cpu()
    H, W = vid_frames.shape[-2:]
    vid_frames = [(haosen.permute(1,2,0).numpy() * 255).astype('uint8') for haosen in vid_frames]
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if pred_masks is not None:
        refer_scores = scores[0] # nq
        max_idx = refer_scores.argmax()
        H, W
        pred_masks = pred_masks.detach().cpu() # nq t h w
        pred_masks = F.interpolate(pred_masks.float(), size=[H, W], mode='bilinear', align_corners=False) > 0
        pred_mask = pred_masks[0] # t h w
        # detectron2
        detectron2_image = torch.stack([torch.from_numpy(generate_instance_canvas_uou(vid_frames=vid_frames[frame_idx], metadata=None, 
                                                            H=pred_mask.shape[1], W=pred_mask.shape[2], pred_mask=pred_mask[frame_idx], color_dataset=color))
                                                    for frame_idx in range(len(pred_mask))], dim=0) # t h w c, uint8
        wt_video = detectron2_image
    else:
        wt_video = torch.stack([torch.from_numpy(f) for f in vid_frames], dim=0)
    if H % 2 != 0:
        H = H - 1
        
    if W % 2 != 0:
        W = W - 1

    write_video(os.path.join(directory, file_name), wt_video[:, :H, :W], fps=fps)

