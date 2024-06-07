import torch
import numpy as np
import random
import math
import logging
import time
import os
from utils.misc import reduce_dict, to_device, is_dist_avail_and_initialized
import gc
from utils.misc import  SmoothedValue, MetricLogger
from torch.nn.parallel import DistributedDataParallel as DDP
import detectron2.utils.comm as comm
import datetime
import torch.distributed as dist
from models import model_entrypoint
from utils.misc import to_device

__all__ = ['Trainer']
class Trainer:
    def __init__(self, configs):
        torch.autograd.set_detect_anomaly(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        seed = configs['model_schedule_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        with torch.cuda.device(self.device):
            torch.cuda.manual_seed(seed)        

        # model and data
        create_model_schedule = model_entrypoint(configs['model']['name'])
        self.model, self.optimizer, self.scheduler, \
            self.train_samplers, self.train_loaders, self.log_lr_group_name_to_idx, \
        self.eval_function = create_model_schedule(configs, device=self.device,)
        self.register_metric_logger([f'lr_group_{haosen}' for haosen in list(self.log_lr_group_name_to_idx.keys())])
        logging.debug(f'total number of parameters:{sum(p.numel() for p in self.model.parameters())}')
        logging.debug(f'total number of trainable parameters:{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        logging.debug(configs)
        self.eval_seed = configs['eval_seed']
        self.out_dir = configs['out_dir']
        self.ckpted_iters = configs['optim']['ckpted_iters'] # list[int]
        self.num_iterations = 0 
        self.num_samples = 0
        assert self.train_samplers[0].start_idx == self.num_samples
        if comm.get_world_size() > 1:
            self.model = DDP(self.model, device_ids=[comm.get_local_rank()], find_unused_parameters=True, broadcast_buffers = False)
    
        random.seed(seed + comm.get_rank())
        np.random.seed(seed + comm.get_rank())
        torch.random.manual_seed(seed + comm.get_rank())
        with torch.cuda.device(self.device):
            torch.cuda.manual_seed(seed + comm.get_rank()) 

        if configs['initckpt']['path'] != '':
            self.load_ckpt(configs['initckpt']['path'], 
                           load_random=configs['initckpt']['load_random'], 
                           load_model=configs['initckpt']['load_model'], 
                           load_schedule=configs['initckpt']['load_schedule'], 
                           load_optimize=configs['initckpt']['load_optimizer'])
        self.save_ckpt() 
        if configs['initckpt']['eval_init_ckpt']:
            self.evaluate() 
            self.load_ckpt(os.path.join(self.iteration_dir, 'ckpt.pth.tar'), 
                       load_random=True, load_schedule=False, load_model=False, load_optimize=False,)

    def train(self):   
        manual_stop_train = False
        for loader in self.train_loaders:
            for idx, batch_dict in enumerate(loader):
                if manual_stop_train:
                    self.save_ckpt()
                self.model.train()
                meta_idxs = batch_dict.pop('meta_idxs')
                visualize = batch_dict.pop('visualize')
                batch_dict = to_device(batch_dict, self.device)
                batch_dict['visualize_paths'] = self.visualize_path(meta_idxs=meta_idxs, 
                                                                    visualize=visualize) 
                iteration_time = time.time()
                loss_dict_unscaled, loss_weight = self.model(batch_dict)
                loss = sum([loss_dict_unscaled[k] * loss_weight[k] for k in loss_weight.keys()])
                assert math.isfinite(loss.item()), f"Loss is {loss.item()}, stopping training"
                loss.backward()       
                self.optimizer.step()
                iteration_time = time.time() - iteration_time
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step() 
                sample_idxs = comm.all_gather(meta_idxs) 
                sample_idxs = [taylor for cardib in sample_idxs for taylor in cardib]
                self.num_samples += len(sample_idxs)
                self.num_iterations += 1
                loss_dict_unscaled_item = {key: torch.tensor(value.detach().item(), device=self.device) for key, value in loss_dict_unscaled.items()}
                del loss, loss_dict_unscaled
                self._log(loss_dict_unscaled=loss_dict_unscaled_item,
                          loss_weight=loss_weight,
                          sample_idxs=sample_idxs,
                          iteration_time=iteration_time)
    def save_ckpt(self):
        rng_state_dict = {comm.get_rank(): {
            'cpu_rng_state': torch.get_rng_state(),
            'gpu_rng_state': torch.cuda.get_rng_state(self.device),
            'numpy_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate()
        }}

        rng_state_dict_by_rank = comm.gather(rng_state_dict, dst=0)

        if comm.is_main_process():
            rng_state_dict_by_rank = {key : value for rs in rng_state_dict_by_rank for key,value in rs.items()}
            model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
            checkpoint_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'num_samples': self.num_samples,
                'num_iterations': self.num_iterations,
                'rng_state_dict_by_rank': rng_state_dict_by_rank, 
                'metrics': {},
            }
            os.makedirs(self.iteration_dir, exist_ok=True)
            torch.save(checkpoint_dict, os.path.join(self.iteration_dir, 'ckpt.pth.tar'))
            del checkpoint_dict
        if is_dist_avail_and_initialized():
            dist.barrier()
        del rng_state_dict_by_rank

    @torch.no_grad()
    def evaluate(self):
        random.seed(self.eval_seed)
        np.random.seed(self.eval_seed)
        torch.random.manual_seed(self.eval_seed)
        with torch.cuda.device(self.device):
            torch.cuda.manual_seed(self.eval_seed)     
        self.model.eval()
        eval_model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_file = os.path.join(self.iteration_dir, 'ckpt.pth.tar')
        assert os.path.exists(ckpt_file), 'must save ckpt before evaluate'
        evaluate_metrics = self.eval_function(model = eval_model,  output_dir = self.iteration_dir)
        if is_dist_avail_and_initialized():
            dist.barrier()
        if comm.is_main_process():
            checkpoint_dict = torch.load(ckpt_file, map_location='cpu')
            ckpt_metrics = checkpoint_dict['metrics']
            to_update_metrics = {}
            for metric_key in evaluate_metrics.keys():
                metric_value = evaluate_metrics[metric_key]
                if metric_key in ckpt_metrics:
                    saved_value = ckpt_metrics[metric_key]
                    if (metric_value - saved_value) > 1e-2:
                        logging.error(f'{metric_key} different saved value')
                        to_update_metrics[metric_key] = metric_value
                else:
                    to_update_metrics[metric_key] = metric_value
            checkpoint_dict['metrics'] = evaluate_metrics
            metric_string = ' '.join([f'{key} : {value:.6f}' for key, value in evaluate_metrics.items()])
            logging.debug(metric_string)
            torch.save(checkpoint_dict, ckpt_file)
            del checkpoint_dict

        if is_dist_avail_and_initialized():
            dist.barrier()

    def load_ckpt(self, 
                  ckpt_path=None, 
                  load_schedule=False,
                  load_optimize=False,  
                  load_model=False,
                  load_random=False, 
                  ):
        assert os.path.exists(ckpt_path)
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if load_model:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
            
        if load_optimize:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        if load_schedule:
            self.num_samples = checkpoint['num_samples'] 
            self.num_iterations = checkpoint['num_iterations'] 

            sampler = self.train_samplers[0]
            while (sampler.end_idx != None) and (self.num_samples > (sampler.end_idx - 1)):
                self.train_samplers.pop(0)
                self.train_loaders.pop(0)
                sampler = self.train_samplers[0]
            self.train_samplers[0].set_iter_first_sample_idx(self.num_samples)

        if load_random:
            rng_state_dict_by_rank = checkpoint['rng_state_dict_by_rank']
            torch.set_rng_state(rng_state_dict_by_rank[comm.get_rank()]['cpu_rng_state'])
            torch.cuda.set_rng_state(rng_state_dict_by_rank[comm.get_rank()]['gpu_rng_state'], device=self.device)
            np.random.set_state(rng_state_dict_by_rank[comm.get_rank()]['numpy_rng_state'])
            random.setstate(rng_state_dict_by_rank[comm.get_rank()]['py_rng_state'])

        del checkpoint

    def _log(self, 
             loss_dict_unscaled,
             loss_weight,
             sample_idxs, 
             iteration_time,):
        loss_dict_unscaled_reduced = reduce_dict(loss_dict_unscaled) 
        loss_value = sum([loss_dict_unscaled_reduced[key] * loss_weight[key] for key in loss_weight.keys()])

        if comm.is_main_process():
            for idx, sp_idx in enumerate(sample_idxs):
                pass
            logger_updates = {}
            for log_lr_group_name, log_lr_group_idx in self.log_lr_group_name_to_idx.items():
                if log_lr_group_idx is None:
                    logger_updates[f'lr_group_{log_lr_group_name}'] = 0
                else:
                    logger_updates[f'lr_group_{log_lr_group_name}'] = self.optimizer.param_groups[log_lr_group_idx]["lr"]
            
            logger_updates.update(loss_dict_unscaled_reduced)
            logger_updates.update({
                'loss_value': loss_value,
                'iteration_time': iteration_time,
            })
            self.metric_logger.update(**logger_updates)
            log_string = self.log_header(iteration_time, sample_idxs) + f'\n{str(self.metric_logger)}'
            wandb_log = self.metric_logger.to_dict()
            logging.debug(log_string)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if type(self.ckpted_iters) == int:
            do_ckpt = (self.num_iterations % self.ckpted_iters) == 0
        elif type(self.ckpted_iters) == list:
            do_ckpt = self.num_iterations in self.ckpted_iters
        else:
            raise ValueError()
        if (self.num_iterations % 2000 == 0) or do_ckpt:
            gc.collect()
            torch.cuda.empty_cache()
        if do_ckpt:
            try: 
                self.save_ckpt() 
                self.evaluate()
                self.load_ckpt(os.path.join(self.iteration_dir, 'ckpt.pth.tar'),  
                            load_random=True, load_schedule=False, load_model=False, load_optimize=False,)
            except:
                if comm.is_main_process():
                    logging.error(f'Iteration {self.num_iterations} evaluate error')
        if is_dist_avail_and_initialized():
            dist.barrier()


    @property
    def device(self):
        return torch.device(comm.get_local_rank())
    
    @property
    def iteration_dir(self):
        return os.path.join(self.out_dir, f'epc[{self.epoch[-1]}]_iter[{self.num_iterations}]_sap[{self.num_samples}]')

    @property
    def epoch(self):
        dataset_length = len(self.train_loaders[0].dataset)
        epoch = self.num_samples / dataset_length
        int_part, dec_part = f'{epoch:.2f}'.split('.')
        return epoch, f'{int_part}_{dec_part}'

    def log_header(self, iteration_time, sample_idxs):
        one_epoch_iterations = len(self.train_loaders[0].dataset) // len(sample_idxs)
        eta = datetime.timedelta(seconds=one_epoch_iterations * iteration_time)
        return f'Epoch_ETA: [{str(eta)}] Epoch:[{self.epoch[0]:.2f}] Iter: [{(self.num_iterations):06d}] Sample: [{self.num_samples:06d}]'
 
    def visualize_path(self, meta_idxs, visualize):
        return [os.path.join(self.iteration_dir, 'visualize_model', f'train_meta_{str(meta_idx)}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]

    def register_metric_logger(self, log_keys):
        if comm.is_main_process():
            if not hasattr(self, 'metric_logger'):
                self.metric_logger = MetricLogger(delimiter='\t')

            for haosen in log_keys:
                if 'lr_group' in haosen:
                    self.metric_logger.add_meter(haosen, SmoothedValue(window_size=1,fmt='{value:.8f}', handler='value'))
                elif haosen == 'iteration_time':
                    self.metric_logger.add_meter(haosen, SmoothedValue(window_size=1,fmt='{value:2f}',handler='value'))
                else:
                    raise ValueError()