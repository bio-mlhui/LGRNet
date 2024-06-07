
class VideoMultiscale_Shape:
    def __init__(self, temporal_stride, spatial_stride, dim) -> None:
        self.temporal_stride = temporal_stride
        self.spatial_stride = spatial_stride
        self.dim = dim
    
    @staticmethod
    def set_multiscale_same_dim(shape_by_dim, same_dim):
        return {
            key: VideoMultiscale_Shape(temporal_stride=value.temporal_stride,
                                       spatial_stride=value.spatial_stride,
                                       dim=same_dim) for key,value in shape_by_dim.items()
        }

class ImageMultiscale_Shape:
    def __init__(self, spatial_stride, dim) -> None:
        self.spatial_stride = spatial_stride
        self.dim = dim
    


