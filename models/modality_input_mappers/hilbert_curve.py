
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
import logging
import torch
from models.layers.gilbert.gilbert2d import gilbert2d_widthBigger
@MODELITY_INPUT_MAPPER_REGISTRY.register()
class HilbertCurve_FrameQuery:
    def __init__(self,
                 configs,
                 ) -> None:
        
        self.frame_query_number = configs['frame_query_number']   
              
    def mapper(self, video):
        return {
            'haosen': None,
        }
        
    def collate(self, list_of_haosen, batch_videos):
        batch_size, T = batch_videos.shape[:2] 
        batch_size, T, _, H, W = batch_videos.shape
        hilbert_curve = list(gilbert2d_widthBigger(width=self.frame_query_number, height=T)) # list[(x(width), y(height))]
        hilbert_curve = torch.tensor(hilbert_curve).long()
        hilbert_curve = hilbert_curve[:, 1] * self.frame_query_number + hilbert_curve[:, 0]
        
        return { 
            'hilbert_curve': hilbert_curve,
        }


        

     