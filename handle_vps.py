
import cv2
import numpy as np
import os
import shutil
from PIL import Image
import torch
from tqdm import tqdm
dataset_root = os.getenv('DATASET_PATH')
# the original IVPS is the union of Kvasir and per-frame Mayo/CVC
all_images = os.listdir(f'{dataset_root}/MICCAI-VPS-dataset/IVPS-TrainSet/Frame')
ka_images = [b for b in all_images if b.startswith('K')]
assert len(ka_images) == 1000

all_gts = os.listdir(f'{dataset_root}/MICCAI-VPS-dataset/IVPS-TrainSet/GT')
ka_gts = [b for b in all_gts if b.startswith('K')]
assert len(ka_gts) == 1000

os.makedirs(os.path.join(f'{dataset_root}/MICCAI-VPS-dataset/Kvasir-SEG/Frame/1'),exist_ok=True)
os.makedirs(os.path.join(f'{dataset_root}/MICCAI-VPS-dataset/Kvasir-SEG/GT/1'),exist_ok=True)

for image_id in tqdm(ka_images):
    shutil.copy(os.path.join(f'{dataset_root}/MICCAI-VPS-dataset/IVPS-TrainSet/Frame', f'{image_id}'),
                os.path.join(f'{dataset_root}/MICCAI-VPS-dataset/Kvasir-SEG/Frame/1', f'{image_id}'),)
    
for image_id in tqdm(ka_gts):
    shutil.copy(os.path.join(f'{dataset_root}/MICCAI-VPS-dataset/IVPS-TrainSet/GT', f'{image_id}'),
                os.path.join(f'{dataset_root}/MICCAI-VPS-dataset/Kvasir-SEG/GT/1', f'{image_id}'),)   

# normalize train directory
for base_path in [f'{dataset_root}/MICCAI-VPS-dataset/VPS-TrainSet/CVC-ColonDB-300/Train',
                  f'{dataset_root}/MICCAI-VPS-dataset/VPS-TrainSet/ASU-Mayo_Clinic/Train',
                  f'{dataset_root}/MICCAI-VPS-dataset/VPS-TrainSet/CVC-ClinicDB-612/Train']:
    video_ids = os.listdir(base_path)
    frame_path = os.path.join(base_path, 'Frame')
    gt_path = os.path.join(base_path, 'GT')
    os.makedirs(frame_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    # Iterate through each video ID directory
    for vid in video_ids:
        shutil.copytree(os.path.join(base_path, vid, 'Frame'), os.path.join(frame_path, vid))
        shutil.copytree(os.path.join(base_path, vid, 'GT'), os.path.join(gt_path, vid))
        # TODO: dangerous: remove if you want

# remove non-mask frames of each training set
SET_NAME = [
    'Kvasir-train',
    'Mayo-train',
    '300-train',
    '612-train',
]

SET_NAME_TO_DIR = {
    'Kvasir-train': 'MICCAI-VPS-dataset/Kvasir-SEG',
    'Mayo-train': 'MICCAI-VPS-dataset/VPS-TrainSet/ASU-Mayo_Clinic/Train',
    '300-train': 'MICCAI-VPS-dataset/VPS-TrainSet/CVC-ColonDB-300/Train',
    '612-train': 'MICCAI-VPS-dataset/VPS-TrainSet/CVC-ClinicDB-612/Train',
}

SET_NAME_TO_NUM_VIDEOS = {
    'Kvasir-train': 1,
    'Mayo-train': 10,
    '300-train': 6,
    '612-train': 18,
    '300-tv': 6,
    '612-test': 5,
    '612-val': 5      
}


SET_NAME_TO_PREFIX = {
    'Kvasir-train': 'Kvasir-train',
    'Mayo-train': 'Mayo-train',
    '300-train': '300-train',
    '612-train': '612-train',
}


root = os.getenv('DATASET_PATH')
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
    
num_delted_frames = 0
for train_set_name in SET_NAME:
    set_dir = SET_NAME_TO_DIR[train_set_name]
    frames_dir = os.path.join(root, set_dir, 'Frame')
    mask_dir = os.path.join(root, set_dir, 'GT')

    video_ids = os.listdir(frames_dir)
    for vid in tqdm(video_ids):
        frames = [haosen[:-4] for haosen in os.listdir(os.path.join(frames_dir, vid))]
        frame_has_fore = [get_frames_mask(mask_dir, vid, [haosen])[0].any() for haosen in tqdm(frames)] # list[t]
        assert len(frame_has_fore) == len(frames)
        num_delted_frames += (~ torch.tensor(frame_has_fore)).int().sum()
        for haosen, frame_name in tqdm(zip(frame_has_fore, frames)):
            if not haosen:
                os.remove(os.path.join(frames_dir, vid, f'{frame_name}.jpg'))

                if os.path.exists(os.path.join(mask_dir, vid, f'{frame_name}.jpg')):
                    os.remove(os.path.join(mask_dir, vid, f'{frame_name}.jpg'))
                elif os.path.exists(os.path.join(mask_dir, vid, f'{frame_name}.png')):
                    os.remove(os.path.join(mask_dir, vid, f'{frame_name}.png')) 
                else:
                    raise ValueError()

print(f'should be {num_delted_frames}/1546.') # should be 1546




