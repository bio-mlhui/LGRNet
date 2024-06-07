
import torch

def bounding_box_from_mask(mask):
    if not mask.any():
        return torch.zeros([4]).float()
    rows = torch.any(mask, dim=1) # h
    cols = torch.any(mask, dim=0) # w
    row_indexs = torch.where(rows)[0]
    rmin, rmax = row_indexs.min(), row_indexs.max()

    col_indexs = torch.where(cols)[0]
    cmin, cmax = col_indexs.min(), col_indexs.max()
    return torch.tensor([cmin, rmin, cmax, rmax]).float() # x1y1x2y2
