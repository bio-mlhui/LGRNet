import torch
from functools import partial
import logging
import numpy as np


def build_scheduler(configs, optimizer):
    name = configs['optim']['scheduler']['name']
    scheduler_configs = configs['optim']['scheduler']

    if name == 'multistep_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=scheduler_configs['milestones'],
                                                        gamma=scheduler_configs['gamma'],
                                                        verbose=scheduler_configs['verbose'],)
        return scheduler
    else:
        raise ValueError()



