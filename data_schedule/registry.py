from detectron2.utils.registry import Registry


EVALUATOR_REGISTRY = Registry('EVALUATOR')
MAPPER_REGISTRY = Registry('MAPPER')

class Mapper:
    def __init__(self, 
                meta_idx_shift,
                dataset_meta,) -> None:
        self.meta_idx_shift = meta_idx_shift 
        self.visualized_meta_idxs = dataset_meta.get('visualize_meta_idxs') 

    def _call(self, data_dict):
        pass

    def __call__(self, data_dict):
        meta_idx = data_dict['meta_idx']
        ret = self._call(data_dict)
        if ret is None:
            return None
        ret['meta_idx'] = meta_idx + self.meta_idx_shift
        if meta_idx in self.visualized_meta_idxs:
            ret['visualize'] = True
        else:
            ret['visualize'] = False
        return ret
    





