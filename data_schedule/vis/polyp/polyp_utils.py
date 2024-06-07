
import os
import numpy as np
import torch
from PIL import Image

SET_NAME = ['polyp_train', 
         'polyp_hard_seen_validate', 
         'polyp_hard_unseen_validate', 
         'polyp_easy_seen_validate', 
         'polyp_easy_unseen_validate',

        'polyp_hard_validate',
        'polyp_easy_validate',
         'Kvasir-train',
         'Mayo-train',
         '300-train',
         '612-train',
         '300-tv',
         '612-test',
         '612-val'
         ]

SET_NAME_TO_DIR = {
    'polyp_train': 'TrainDataset',
    'polyp_hard_seen_validate': 'TestHardDataset/Seen',
    'polyp_hard_unseen_validate': 'TestHardDataset/Unseen',
    'polyp_easy_seen_validate': 'TestEasyDataset/Seen',
    'polyp_easy_unseen_validate': 'TestEasyDataset/Unseen',
    
    'polyp_hard_validate': 'TestHardDataset/Combine',
    'polyp_easy_validate': 'TestEasyDataset/Combine',
    
    'Kvasir-train': 'MICCAI-VPS-dataset/Kvasir-SEG',
    'Mayo-train': 'MICCAI-VPS-dataset/VPS-TrainSet/ASU-Mayo_Clinic/Train',
    '300-train': 'MICCAI-VPS-dataset/VPS-TrainSet/CVC-ColonDB-300/Train',
    '612-train': 'MICCAI-VPS-dataset/VPS-TrainSet/CVC-ClinicDB-612/Train',
    '300-tv': 'MICCAI-VPS-dataset/VPS-TestSet/CVC-ColonDB-300',
    '612-test': 'MICCAI-VPS-dataset/VPS-TestSet/CVC-ClinicDB-612-Test',
    '612-val': 'MICCAI-VPS-dataset/VPS-TestSet/CVC-ClinicDB-612-Valid'
}

SET_NAME_TO_NUM_VIDEOS = {
    'polyp_train': 112,
    'polyp_hard_seen_validate': 17,
    'polyp_hard_unseen_validate': 37,
    'polyp_easy_seen_validate': 33,
    'polyp_easy_unseen_validate': 86,
    'polyp_hard_validate': 54,
    'polyp_easy_validate': 119,
    'Kvasir-train': 1,
    'Mayo-train': 10,
    '300-train': 6,
    '612-train': 18,
    '300-tv': 6,
    '612-test': 5,
    '612-val': 5      
}

SET_NAME_TO_MODE = {
    'polyp_train': 'train',
    'polyp_hard_seen_validate': 'evaluate',
    'polyp_hard_unseen_validate': 'evaluate',
    'polyp_easy_seen_validate': 'evaluate',
    'polyp_easy_unseen_validate': 'evaluate',
    'polyp_hard_validate': 'evaluate',
    'polyp_easy_validate': 'evaluate',
    'Kvasir-train': 'train',
    'Mayo-train': 'train',
    '300-train': 'train',
    '612-train': 'train',
    '300-tv': 'evaluate',
    '612-test': 'evaluate',
    '612-val': 'evaluate'        
}

SET_NAME_TO_PREFIX = {
    'polyp_train': 'polyp_train',
    'polyp_hard_seen_validate': 'polyp_hard_seen_validate',
    'polyp_hard_unseen_validate': 'polyp_hard_unseen_validate',
    'polyp_easy_seen_validate': 'polyp_easy_seen_validate',
    'polyp_easy_unseen_validate': 'polyp_easy_unseen_validate',
    'polyp_hard_validate': 'polyp_hard_validate',
    'polyp_easy_validate': 'polyp_easy_validate',
    'Kvasir-train': 'Kvasir-train',
    'Mayo-train': 'Mayo-train',
    '300-train': '300-train',
    '612-train': '612-train',
    '300-tv': '300-tv',
    '612-test': '612-test',
    '612-val': '612-val'  
}

CLASS_TO_ID = {
    'high_grade_adenoma':0, 
    'hyperplastic_polyp':1, 
    'invasive_cancer':2,
    'low_grade_adenoma':3, 
    'sessile_serrated_lesion':4,
    'traditional_serrated_adenoma':5
}

def get_frames(frames_path, video_id, frames):
    return [Image.open(os.path.join(frames_path, video_id, f'{f}.jpg')).convert('RGB') for f in frames]
def get_frames_mask(mask_path, video_id, frames):
    # masks = [cv2.imread(os.path.join(mask_path, video_id, f'{f}.jpg')) for f in frames]
    if os.path.exists(os.path.join(mask_path, video_id, f'{frames[0]}.png')):
        masks = [Image.open(os.path.join(mask_path, video_id, f'{f}.png')).convert('L') for f in frames]
    elif os.path.exists(os.path.join(mask_path, video_id, f'{frames[0]}.jpg')):
        masks = [Image.open(os.path.join(mask_path, video_id, f'{f}.jpg')).convert('L') for f in frames]
    else:
        raise ValueError()
    masks = [np.array(mk) for mk in masks]
    masks = torch.stack([torch.from_numpy(mk) for mk in masks], dim=0) # t h w
    # assert set(masks.unique().tolist()) == set([0, 255]), f'{masks.unique().tolist()}'
    masks = (masks > 0).int()
    return masks, torch.ones(len(frames)).bool()


