import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler

from net.dataset.dataset import DoughDataset
from net.pipeline.pipeline import Pipeline

import torch.distributed as dist
import os


def init_ddp(backend='nccl'):
    dist.init_process_group(backend, rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))

def get_dataset(config, ddp=False):
    dataset = DoughDataset(config)
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=(config.subset == 'train'))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.bs,
                                                 drop_last=(config.subset != 'test'),
                                                 sampler=sampler,
        )
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.bs,
                                                 shuffle=(config.subset == 'train'),
                                                 drop_last=(config.subset != 'test'),
    )
    return dataset, dataloader, sampler

def get_model(config, ddp=False):
    model = Pipeline(config)
    if ddp:
        local_rank = int(os.environ['LOCAL_RANK'])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)

    return model

def get_optimizer(config, model, parameters=None, names=None):
    if config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            name_param = zip(names, parameters) if names is not None else model.named_parameters()
            for name, param in name_param:
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'z_' in name or name in skip_list:
                    # print(f'excluding {name} from weight decay')
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(model, weight_decay=config.kwargs.weight_decay / config.kwargs.lr)
        optimizer = optim.AdamW(param_groups, **config.kwargs)
    elif config.type == 'Adam':
        optimizer = optim.Adam(parameters, **config.kwargs)
    elif config.type == 'SGD':
        optimizer = optim.SGD(parameters, nesterov=True, **config.kwargs)
    else:
        raise NotImplementedError()
    
    return optimizer

def get_scheduler(config, optimizer):
    if config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=config.kwargs.epochs,
                t_in_epochs=True,
                cycle_mul=1,
                cycle_decay=0.1,
                cycle_limit=1,
                lr_min=config.kwargs.lr_min,
                warmup_lr_init=config.kwargs.warmup_lr_init,
                warmup_t=config.kwargs.initial_epochs,)
    else:
        raise NotImplementedError()
    
    return scheduler
