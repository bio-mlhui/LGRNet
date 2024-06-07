
import os

if os.getenv('CURRENT_TASK') == 'VIS':
    from . import vis
else:
    raise ValueError()

def build_schedule(configs, model_input_mapper, model_input_collate_fn):
    import logging
    from functools import partial
    import detectron2.utils.comm as comm
    from torch.utils.data import DataLoader, ConcatDataset
    from .registry import MAPPER_REGISTRY, EVALUATOR_REGISTRY
    from detectron2.data import DatasetCatalog, DatasetFromList, MapDataset, MetadataCatalog
    from data_schedule.utils.sampler import Evaluate_ExactSampler_Distributed, Train_InfiniteSampler_Distributed
    datasets = {'train': [], 'evaluate': []}
    meta_idx_shift = 0 
    for mode in ['train', 'evaluate']:
        for dataset_name in configs['data'][mode].keys():
            dataset_assume_mode = MetadataCatalog.get(dataset_name).get('mode')
            if dataset_assume_mode != mode:
                logging.warning(f'default mode of {dataset_name} is {dataset_assume_mode} not {mode}')
            dataset_dicts = DatasetFromList(DatasetCatalog.get(dataset_name), copy=True, serialize=True)
            mapper = MAPPER_REGISTRY.get(configs['data'][mode][dataset_name]['mapper']['name'])(mode=mode,
                                                                                                dataset_name=dataset_name, 
                                                                                                configs=configs,
                                                                                                meta_idx_shift=meta_idx_shift if mode == 'train' else 0)
            meta_idx_shift += len(dataset_dicts)
            dataset = MapDataset(dataset_dicts, partial(composition, mappers=[mapper, 
                                                                              partial(model_input_mapper, mode=mode)]))
            if mode == 'train':
                datasets[mode].append(dataset)
            else:
                datasets[mode].append((dataset_name, dataset))

    train_dataset = ConcatDataset(datasets['train'])
    logging.debug(f'Total number of training meta: {len(train_dataset)}')

    train_loader_splits = configs['optim']['splits']
    batch_sizes = configs['optim']['batch_sizes']
    splits = list(zip(train_loader_splits[:-1], train_loader_splits[1:]))
    assert len(splits) == (len(batch_sizes))
    inf_stream_fn = partial(infinite_indices,
                            seed=configs['stream_idx_seed'], 
                            batch_sizes=configs['optim']['batch_sizes'],
                            splits=configs['optim']['splits'],
                            one_batch_two_epoch=configs['optim']['one_batch_two_epoch'],
                            dataset_length=len(train_dataset),
                            shuffle=True) 
    train_samplers = []
    train_loaders = []
    for btch_size, (range_start, range_end) in zip(batch_sizes, splits):
        if range_end is not None:
            assert (range_end - range_start) % btch_size == 0, ''
        assert btch_size % comm.get_world_size() == 0, ''
        each_process_batch_size = int(btch_size / comm.get_world_size())
        loader_sampler = Train_InfiniteSampler_Distributed(inf_stream_fn=inf_stream_fn,
                                                           start_idx=range_start,
                                                           end_idx=range_end,)
        train_samplers.append(loader_sampler)
        train_loaders.append(DataLoader(train_dataset,
                                        batch_size=each_process_batch_size,
                                        sampler=loader_sampler,
                                        collate_fn=partial(model_input_collate_fn, mode='train'), 
                                        num_workers=int(os.getenv('TORCH_NUM_WORKERS')),
                                        pin_memory=True,
                                        persistent_workers=True))

    evaluators = []
    for eval_dataset_name, eval_dataset in datasets['evaluate']:
        logging.debug(f'Number of evaluate meta in {eval_dataset_name}: {len(eval_dataset)}')
        loader = DataLoader(eval_dataset, 
                            batch_size=1, 
                            sampler=Evaluate_ExactSampler_Distributed(eval_dataset),
                            collate_fn=partial(model_input_collate_fn, mode='evaluate'),
                            num_workers=int(os.getenv('TORCH_NUM_WORKERS')),
                            pin_memory=True,
                            persistent_workers=True)
        
        evaluator = EVALUATOR_REGISTRY.get(configs['data']['evaluate'][eval_dataset_name]['evaluator']['name'])(configs=configs,
                                                                                                                dataset_name=eval_dataset_name,
                                                                                                                data_loader=loader)
        evaluators.append((eval_dataset_name, evaluator))

    return train_samplers, train_loaders, partial(evaluate_call, evaluators=evaluators)

def composition(data_dict, mappers):
    for mappper in mappers:
        data_dict = mappper(data_dict)
        if data_dict is None:
            return None
    return data_dict

def evaluate_call(evaluators, model, output_dir):
    import detectron2.utils.comm as comm
    ret = {}
    for eval_dataset_name, evaluator in evaluators:
        metric_dict = evaluator(model=model,output_dir=output_dir)
        if comm.is_main_process():
            for key, value in metric_dict.items():
                assert f'{key}_{eval_dataset_name}' not in ret
                ret[f'{key}_{eval_dataset_name}'] = value
        comm.synchronize()
    return ret


def _infinite_indices(seed, dataset_length, shuffle=True,):
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    while True:
        if shuffle:
            yield from torch.randperm(dataset_length, generator=g).tolist()
        else:
            yield from torch.arange(dataset_length).tolist()

def infinite_indices(seed, 
                     dataset_length, 
                     batch_sizes, 
                     splits, 
                     one_batch_two_epoch='just_use',
                     shuffle=True): # 'abandon', 'just_use', 'pad'
    import torch
    import math
    g = torch.Generator()
    g.manual_seed(seed)

    split_ranges = list(zip(splits[:-1], splits[1:]))
    assert len(split_ranges) == (len(batch_sizes))
    stream = _infinite_indices(seed, dataset_length=dataset_length, shuffle=shuffle)

    stream_throw_cnt = 0
    cnt = 0
    for (range_start, range_end), btch_size in zip(split_ranges, batch_sizes):
        assert cnt == range_start
        if range_end == None:
            range_end = math.inf
        
        while cnt < range_end:
            epoch_milestone = ((stream_throw_cnt // dataset_length) + 1 ) * dataset_length
            if (stream_throw_cnt < epoch_milestone) and (stream_throw_cnt + btch_size > epoch_milestone) and (one_batch_two_epoch != 'just_use'):
                if one_batch_two_epoch == 'abandon':
                    for _ in range(epoch_milestone - stream_throw_cnt):
                        abandon = next(stream)
                        stream_throw_cnt += 1

                elif one_batch_two_epoch == 'pad':
                    diff = stream_throw_cnt + btch_size - epoch_milestone
                    num_throw = btch_size - diff
                    rand_idxs = torch.randperm(dataset_length, generator=g)[:diff].tolist()
                    for _ in range(num_throw):
                        cnt += 1
                        stream_throw_cnt += 1
                        yield next(stream)
                    for idx in rand_idxs:
                        cnt += 1
                        yield idx
                else:
                    raise ValueError()
            else:
                for _ in range(btch_size):
                    cnt += 1
                    stream_throw_cnt += 1
                    yield next(stream)  
 
        assert cnt == range_end
