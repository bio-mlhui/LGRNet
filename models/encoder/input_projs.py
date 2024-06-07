
import torch.nn as nn
from detectron2.modeling import META_ARCH_REGISTRY
from models.layers.anyc_trans import Linear_NormAct

from models.layers.anyc_trans import Conv3d_NormAct, Conv2d_NormAct
from einops import rearrange

@META_ARCH_REGISTRY.register()
class VideoConv3d_TextLinear(nn.Module):
    """
    如果multiscale_shapes是None, 那么每个multiscale_shape的input_dim都是out_dim
    如果multiscale_shapes给出了, 那么按照multiscale shapes里的dim
    """
    def __init__(self,
                 configs,
                 out_dim,
                 text_dim=None, # 如果是none的话, 那么假设等于out_dim
                 multiscale_shapes=None, # scale_name: (dim, [temporal_scale, spatial_scale])
                 ) -> None:
        super().__init__()
        text_dim = out_dim if text_dim is None else text_dim

        multiscale_projs_config = configs['video_multiscale_projs']
        proj_names = multiscale_projs_config.keys() # list[str]

        in_dims = {}
        if multiscale_shapes is not None:
            assert set(proj_names).issubset(set(list(multiscale_shapes.keys())))
            for name in proj_names:
                in_dims[name] = multiscale_shapes[name].dim
        else:
            for name in proj_names:
                in_dims[name] = out_dim

        projs = {}
        for name, config in multiscale_projs_config.items():
            projs[name] = Conv3d_NormAct(in_channels=in_dims[name],
                                         out_channels=out_dim, 
                                         **config)
            
        self.video_multiscale_projs = nn.ModuleDict(projs)

        text_proj_config = configs['text_proj']
        if text_proj_config is None:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = Linear_NormAct(in_features=text_dim, out_features=out_dim, **text_proj_config)

    def forward(self, multiscales, text_dict):
        ret = {}
        for scale_name, scale_feat in multiscales.items(): # b c t h w
            if scale_name in self.video_multiscale_projs:
                scale_feat = self.video_multiscale_projs[scale_name](scale_feat)
                ret[scale_name] = scale_feat
            else:
                ret[scale_name] = scale_feat
        
        if isinstance(text_dict, AMRData):
            text_dict.amr_feats = self.text_proj(text_dict.amr_feats)
            text_dict.text_feats = self.text_proj(text_dict.text_feats)
        else:
            raise ValueError()
        
        return ret, text_dict



@META_ARCH_REGISTRY.register()
class VideoConv2d_TextLinear(nn.Module):
    """
    如果multiscale_shapes是None, 那么每个multiscale_shape的input_dim都是out_dim
    如果multiscale_shapes给出了, 那么按照multiscale shapes里的dim
    """
    def __init__(self,
                 configs,
                 out_dim,
                 text_dim=None, # 如果是none的话, 那么假设等于out_dim
                 multiscale_shapes=None, # scale_name: (dim, [temporal_scale, spatial_scale])
                 ) -> None:
        super().__init__()
        text_dim = out_dim if text_dim is None else text_dim
        multiscale_projs_config = configs['video_multiscale_projs']
        proj_names = multiscale_projs_config.keys() # list[str]

        in_dims = {}
        if multiscale_shapes is not None:
            assert set(proj_names).issubset(set(list(multiscale_shapes.keys())))
            for name in proj_names:
                in_dims[name] = multiscale_shapes[name].dim
        else:
            for name in proj_names:
                in_dims[name] = out_dim

        projs = {}
        for name, config in multiscale_projs_config.items():
            projs[name] = Conv2d_NormAct(in_channels=in_dims[name],
                                         out_channels=out_dim, 
                                         **config)
            
        self.video_multiscale_projs = nn.ModuleDict(projs)

        text_proj_config = configs['text_proj']
        if text_proj_config is None:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = Linear_NormAct(in_features=text_dim, out_features=out_dim, **text_proj_config)

    def forward(self, multiscales, text_dict):
        ret = {}
        for scale_name, scale_feat in multiscales.items(): # b c t h w
            if scale_name in self.video_multiscale_projs:
                batch_size, _, nf = scale_feat.shape[:3]
                scale_feat = rearrange(scale_feat, 'b c t h w -> (b t) c h w')
                scale_feat = self.video_multiscale_projs[scale_name](scale_feat)
                scale_feat = rearrange(scale_feat, '(b t) c h w -> b c t h w', b=batch_size, t=nf)
                ret[scale_name] = scale_feat
            else:
                ret[scale_name] = scale_feat
        
        if isinstance(text_dict, AMRData):
            text_dict.amr_feats = self.text_proj(text_dict.amr_feats)
            text_dict.text_feats = self.text_proj(text_dict.text_feats)
        else:
            raise ValueError()
        
        return ret, text_dict


@META_ARCH_REGISTRY.register()
class ImageConv_MultiscaleProj(nn.Module):
    def __init__(self,
                 configs,
                 out_dim,
                 multiscale_shapes=None,
                 ) -> None:
        """
        如果multiscale_shape是空, 那么输入的dim = out_dim
        """
        super().__init__()
        projs_configs = configs['projs']
        proj_names = list(projs_configs.keys()) # list[str]

        in_dims = {}
        if multiscale_shapes is not None:
            assert set(proj_names).issubset(set(list(multiscale_shapes.keys())))
            for name in proj_names:
                in_dims[name] = multiscale_shapes[name].dim
        else:
            for name in proj_names:
                in_dims[name] = out_dim

        projs = {}
        for name, config in projs_configs.items():
            projs[name] = Conv2d_NormAct(in_channels=in_dims[name], out_channels=out_dim, 
                                         **config)
        self.multiscale_projs = nn.ModuleDict(projs)

    def forward(self, multiscales):
        ret = {}
        for scale_name, scale_feat in multiscales.items():
            if scale_name in self.multiscale_projs:
                scale_feat = self.multiscale_projs[scale_name](scale_feat)
                ret[scale_name] = scale_feat
            else:
                ret[scale_name] = scale_feat
        return ret


@META_ARCH_REGISTRY.register()
class Video2D_ImageConv_MultiscaleProj(nn.Module):
    def __init__(self,
                 configs,
                 out_dim,
                 multiscale_shapes=None, # scale_name: (dim, [temporal_scale, spatial_scale])
                 ) -> None:
        super().__init__()
        self.image_homo = ImageConv_MultiscaleProj(configs=configs, out_dim=out_dim, multiscale_shapes=multiscale_shapes)
    
    def forward(self, multiscales):
        batch_sisze, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]
        # b c t h w -> bt c h w
        multiscales = {key: value.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous() for key,value in multiscales.items()}
        multiscales = self.image_homo(multiscales)
        multiscales = {key: rearrange(value, '(b t) c h w -> b c t h w',b=batch_sisze, t=nf).contiguous()\
                        for key,value in multiscales.items()}
        return multiscales        

@META_ARCH_REGISTRY.register()
class VideoConv_MultiscaleProj(nn.Module):
    def __init__(self,
                 configs,
                 out_dim,
                 multiscale_shapes=None,
                 ) -> None:
        """
        如果multiscale_shape是空, 那么输入的dim = out_dim
        """
        super().__init__()
        projs_configs = configs['projs']
        proj_names = list(projs_configs.keys()) # list[str]

        in_dims = {}
        if multiscale_shapes is not None:
            assert set(proj_names).issubset(set(list(multiscale_shapes.keys())))
            for name in proj_names:
                in_dims[name] = multiscale_shapes[name].dim
        else:
            for name in proj_names:
                in_dims[name] = out_dim

        projs = {}
        for name, config in projs_configs.items():
            projs[name] = Conv3d_NormAct(in_channels=in_dims[name], out_channels=out_dim, 
                                         **config)
        self.multiscale_projs = nn.ModuleDict(projs)

    def forward(self, multiscales):
        ret = {}
        for scale_name, scale_feat in multiscales.items():
            if scale_name in self.multiscale_projs:
                scale_feat = self.multiscale_projs[scale_name](scale_feat)
                ret[scale_name] = scale_feat
            else:
                ret[scale_name] = scale_feat
        return ret

     




@META_ARCH_REGISTRY.register()
class FrameQueryLinear_TextLinear(nn.Module):
    def __init__(self,
                 configs,
                 out_dim,
                 text_dim=None, # int
                 query_dim=None, # scale_name: (dim, [temporal_scale, spatial_scale])
                 ) -> None:
        super().__init__()
        query_proj_config = configs['query_proj']
        query_dim = out_dim if query_dim is None else query_dim
        text_dim = out_dim if text_dim is None else text_dim

        self.query_proj = Linear_NormAct(in_features=query_dim, out_features=out_dim, **query_proj_config)
        text_proj_config = configs['text_proj']
        if text_proj_config is None:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = Linear_NormAct(in_features=text_dim, out_features=out_dim, **text_proj_config)

    def forward(self, frame_query, text_dict):
        # b T nqf c
        # text_dict
        frame_query = self.query_proj(frame_query)

        if isinstance(text_dict, AMRData):
            text_dict.amr_feats = self.text_proj(text_dict.amr_feats)
            text_dict.text_feats = self.text_proj(text_dict.text_feats)
        else:
            raise ValueError()
        
        return frame_query, text_dict


@META_ARCH_REGISTRY.register()
class VideoConv3d_FrameQueryLinear_TextLinear(nn.Module):
    def __init__(self,
                 configs,
                 out_dim,
                 feat_dim=None,
                 text_dim=None, # int
                 query_dim=None, # scale_name: (dim, [temporal_scale, spatial_scale])
                 ) -> None:
        super().__init__()
        query_proj_config = configs['query_proj']
        feat_proj_config = configs['feat_proj']
        text_proj_config = configs['text_proj']

        feat_dim = out_dim if feat_dim is None else feat_dim
        query_dim = out_dim if query_dim is None else query_dim
        text_dim = out_dim if text_dim is None else text_dim

        self.query_proj = Linear_NormAct(in_features=query_dim, out_features=out_dim, **query_proj_config) if query_proj_config is not None else nn.Identity()
        self.text_proj = Linear_NormAct(in_features=text_dim, out_features=out_dim, **text_proj_config) if text_proj_config is not None else nn.Identity()
        self.feat_proj = Conv3d_NormAct(in_channels=feat_dim, out_channels=out_dim, **feat_proj_config)

    def forward(self, mask_feat, frame_query, text_dict):
        mask_feat = self.feat_proj(mask_feat)
        frame_query = self.query_proj(frame_query)
        if isinstance(text_dict, AMRData):
            text_dict.amr_feats = self.text_proj(text_dict.amr_feats)
            text_dict.text_feats = self.text_proj(text_dict.text_feats)
        else:
            raise ValueError()

        return mask_feat, frame_query, text_dict

# 每一个module应该都把input进行一边proj, proj到自己的空间里
@META_ARCH_REGISTRY.register()
class VideoConv3d_FrameQueryLinear(nn.Module):
    """
    如果multiscale_shapes是None, 那么每个multiscale_shape的input_dim都是out_dim
    如果multiscale_shapes给出了, 那么按照multiscale shapes里的dim
    """
    def __init__(self,
                 configs,
                 out_dim,
                 query_dim=None, # 如果是none的话, 那么假设等于out_dim
                 multiscale_shapes=None, # scale_name: (dim, [temporal_scale, spatial_scale])
                 ) -> None:
        super().__init__()
        query_dim = out_dim if query_dim is None else query_dim

        multiscale_projs_config = configs['video_multiscale_projs']
        proj_names = multiscale_projs_config.keys() # list[str]

        in_dims = {}
        if multiscale_shapes is not None:
            assert set(proj_names).issubset(set(list(multiscale_shapes.keys())))
            for name in proj_names:
                in_dims[name] = multiscale_shapes[name].dim
        else:
            for name in proj_names:
                in_dims[name] = out_dim

        projs = {}
        for name, config in multiscale_projs_config.items():
            projs[name] = Conv3d_NormAct(in_channels=in_dims[name],
                                         out_channels=out_dim, 
                                         **config)
            
        self.video_multiscale_projs = nn.ModuleDict(projs)

        query_proj_config = configs['query_proj']
        if query_proj_config is None:
            self.query_proj = nn.Identity()
        else:
            self.query_proj = Linear_NormAct(in_features=query_dim, out_features=out_dim, **query_proj_config)

    def forward(self, multiscales, frame_queries):
        # b t nq c
        ret = {}
        for scale_name, scale_feat in multiscales.items(): # b c t h w
            if scale_name in self.video_multiscale_projs:
                scale_feat = self.video_multiscale_projs[scale_name](scale_feat)
                ret[scale_name] = scale_feat
            else:
                ret[scale_name] = scale_feat
        
        frame_queries = self.query_proj(frame_queries)
        return ret, frame_queries



@META_ARCH_REGISTRY.register()
class FrameQueryLinear(nn.Module):
    def __init__(self,
                 configs,
                 out_dim,
                 query_dim=None, # scale_name: (dim, [temporal_scale, spatial_scale])
                 ) -> None:
        super().__init__()
        query_proj_config = configs['query_proj']
        query_dim = out_dim if query_dim is None else query_dim

        self.query_proj = Linear_NormAct(in_features=query_dim, out_features=out_dim, **query_proj_config)

    def forward(self, frame_query):
        # b T nqf c
        frame_query = self.query_proj(frame_query)
        
        return frame_query

