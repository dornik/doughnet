from colorama import Fore, Style
import numpy as np
import torch
from tqdm import tqdm
import os
import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from net.pipeline.builder import get_dataset, get_model, get_optimizer, get_scheduler, init_ddp
from torch.utils.data import default_collate as collate
from net.pipeline.random import get_state, set_state, manual_seed

DEBUG = hasattr(sys, 'gettrace') and (sys.gettrace() is not None)


def reduce_tensor(tensor, world_size):
    # for acc kind, get the mean in each gpu
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


def custom_collate(all_data):
    if len(all_data[0]['obj_observed'].shape) == 3:
        ks = all_data[0].keys()
        data = {}
        for k in ks:
            data[k] = torch.cat([d[k] for d in all_data], dim=0)
    else:
        data = collate(all_data)
    return data


class PredictionWorkspace:

    def __init__(self, cfg: OmegaConf):
        if cfg.settings.ddp and not DEBUG and not cfg.settings.test_only:
            cfg.cuda_id = int(os.environ['LOCAL_RANK'])  # set cuda id to rank
            init_ddp()
            cfg.cuda_id = torch.distributed.get_rank()  # set cuda id to rank

            self.use_ddp = True
            self.is_main_proc = int(os.environ['LOCAL_RANK']) == 0
        else:
            self.use_ddp = False
            self.is_main_proc = True
        # cuda / used gpu
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.cuda_id)  # not working, see https://github.com/pytorch/pytorch/issues/80876
        torch.cuda.set_device(cfg.cuda_id)  # this is discouraged - but works

        # replaces all macros with concrete value
        OmegaConf.resolve(cfg)
        self.cfg = cfg

        # wandb
        self.wandb = wandb.init(
            config=OmegaConf.to_container(self.cfg, resolve=True),
            **self.cfg.logging
        ) if self.cfg.log_wandb and not DEBUG and self.is_main_proc else None

    @property
    def output_dir(self):
        return HydraConfig.get().runtime.output_dir

    def forward(self, model, data, z=None, val_test=False):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda()

        results, data = model(data, z, val_test)
        return *results, data
    
    def backward(self, model, optimizer, loss):
        loss.backward()
        if self.cfg.training.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.grad_norm_clip, norm_type=2)
        optimizer.step()
        model.zero_grad()

    def log(self, epoch, log_dict,
            train_loss, train_acc, train_loss_dict, train_acc_dict,
            val_loss, val_acc, val_loss_dict, val_acc_dict,
            val_cont_loss, val_cont_acc, val_cont_loss_dict, val_cont_acc_dict,):
        if not self.is_main_proc:
            return
        log_str = f'{Fore.WHITE}end of epoch {epoch+1}: '
        log_str += f'{Fore.CYAN}training loss={train_loss:.3f}'
        if self.cfg.settings.val_single:
            log_str += f' --- {Fore.MAGENTA}validation acc={val_acc*100:0.1f}%, loss={val_loss:.3f}'
        if self.cfg.settings.val_multi:
            log_str += f' --- {Fore.GREEN}continuous acc={val_cont_acc*100:0.1f}%, loss={val_cont_loss:.3f}'
        print(log_str + '\n' + Style.RESET_ALL)
        if self.wandb is not None:
            log_dict['train/loss'] = train_loss
            # log_dict['train/acc'] = train_acc  # not computed, so always returns -1
            if self.cfg.settings.val_single:
                log_dict['val/loss'] = val_loss
                log_dict['val/acc'] = val_acc
            if self.cfg.settings.val_multi:
                log_dict['val-cont/loss'] = val_cont_loss
                log_dict['val-cont/acc'] = val_cont_acc
            for loss_dict, prefix, postfix in zip([train_loss_dict, val_loss_dict, val_cont_loss_dict,
                                                   train_acc_dict, val_acc_dict, val_cont_acc_dict],
                                                  ['train', 'val', 'val-cont']*2,
                                                  ['losses']*3 + ['accs']*3):
                if loss_dict is None:
                    continue
                for k, v in loss_dict.items():
                    log_dict[f'{prefix}-{postfix}/{k}'] = v
            self.wandb.log(log_dict)

    def train(self, model, optimizer, train_dataloader):
        print(f'{Fore.CYAN}', end='')
        model.train()

        losses, accs = [], []
        losses_dict, accs_dict = {}, {}
        for data in tqdm(train_dataloader):
            loss, acc, loss_dict, acc_dict, _ = self.forward(model, data)
            if self.use_ddp:
                loss = reduce_tensor(loss, int(os.environ['WORLD_SIZE']))
                acc = reduce_tensor(acc, int(os.environ['WORLD_SIZE']))
                for k, v in loss_dict.items():
                    loss_dict[k] = reduce_tensor(v, int(os.environ['WORLD_SIZE']))
                for k, v in acc_dict.items():
                    acc_dict[k] = reduce_tensor(v, int(os.environ['WORLD_SIZE']))
                torch.cuda.synchronize()
            self.backward(model, optimizer, loss)

            losses += [float(loss)]
            accs += [float(acc)]
            for k, v in loss_dict.items():
                losses_dict[k] = losses_dict.get(k, []) + [float(v)]
            for k, v in acc_dict.items():
                accs_dict[k] = accs_dict.get(k, []) + [float(v)]
        
        for k, v in losses_dict.items():
            losses_dict[k] = np.mean(v)
        for k, v in accs_dict.items():
            accs_dict[k] = np.mean(v)

        print(f'{Fore.RESET}', end='')
        losses = np.where(np.isnan(losses), 0, losses)
        accs = np.where(np.isnan(accs), 0, accs)
        return np.mean(losses), np.mean(accs), losses_dict, accs_dict
    
    def test(self, model, test_dataloader, continuous=False):
        print(f'{Fore.MAGENTA}', end='')
        if continuous:
            assert self.cfg.dataset.val.bs == 1
        model.eval()
        random_state = get_state()

        z = None
        losses, accs = [], []
        losses_dict, accs_dict, last_accs_dict = {}, {}, {}
        with torch.no_grad():
            cur_scene = -1
            for data in tqdm(test_dataloader):
                if continuous and data['scene'][0] != cur_scene:  # start of a new scene
                    z = None
                    cur_scene = data['scene'][0]
                    if continuous:  # last-frame accuracy
                        for k, v in accs_dict.items():
                            last_accs_dict[f'{k}_last'] = last_accs_dict.get(f'{k}_last', []) + [v[-1]]

                loss, acc, loss_dict, acc_dict, data = self.forward(model, data, z, val_test=True)

                if continuous:
                    z = data['z_pre_nxt'].view(1, -1, self.cfg.dimensions.latent_dim)

                losses += [float(loss)]
                accs += [float(acc)]
                for k, v in loss_dict.items():
                    losses_dict[k] = losses_dict.get(k, []) + [float(v)]
                for k, v in acc_dict.items():
                    accs_dict[k] = accs_dict.get(k, []) + [float(v)]
        # last frame of last scene
        if continuous:
            for k, v in accs_dict.items():
                last_accs_dict[f'{k}_last'] = last_accs_dict.get(f'{k}_last', []) + [v[-1]]
            # assert len(last_accs_dict['pre_viou_nxt_last']) == test_dataloader.dataset.data['scene'].shape[0]  # one per scene
            accs_dict.update(last_accs_dict)
        # mean accuracies over all scenes/frames
        for k, v in losses_dict.items():
            losses_dict[k] = np.mean(v)
        for k, v in accs_dict.items():
            accs_dict[k] = np.mean(v)

        # --

        set_state(random_state)  # continue from previous state
        print(f'{Fore.RESET}', end='')
        losses = np.where(np.isnan(losses), 0, losses)
        accs = np.where(np.isnan(accs), 0, accs)
        return np.mean(losses), np.mean(accs), losses_dict, accs_dict

    def load(self, model, path=None, best=True, pop_condition=True):
        if path is None:
            path = os.path.join(self.output_dir, 'best.pth' if best else 'last.pth')
        infos = torch.load(path, map_location=f'cuda:{self.cfg.cuda_id}')
        state_dict = infos['model_state_dict']
        if pop_condition:
            _ = [state_dict.pop(k) for k in list(state_dict.keys())
                 if k.startswith('model.condition')
                 or k.startswith('module.model.condition')]
        if self.cfg.settings.test_only or not self.use_ddp:
            # fix ddp artefact
            valid_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = '.'.join(k.split('.')[1:])
                valid_state_dict[k] = v
        else:
            valid_state_dict = state_dict
        model.load_state_dict(valid_state_dict, strict=False)

    def save(self, model, val_acc, best_acc):
        if not self.is_main_proc:
            return best_acc
        infos = {
            'model_state_dict': model.state_dict(),
        }
        torch.save(infos, os.path.join(self.output_dir, 'last.pth'))
        if val_acc > best_acc:
            torch.save(infos, os.path.join(self.output_dir, 'best.pth'))
            best_acc = val_acc
        return best_acc
    
    def run(self):
        # reproducability
        torch.backends.cudnn.deterministic = not self.cfg.settings.test_only
        torch.backends.cudnn.benchmark = self.cfg.settings.test_only
        seed = self.cfg.training.seed
        manual_seed(seed)  # sets random, np.random and torch.manual_seed
        # model
        model = get_model(self.cfg, self.use_ddp)

        if not self.cfg.settings.test_only:

            # load pretrained model
            if self.cfg.settings.resume:
                self.load(model, path=self.cfg.settings.resume_path, pop_condition=True)
                if self.cfg.settings.resume_freeze:
                    # freeze everything but condition
                    parameters, names = [], []
                    for name, param in model.named_parameters():
                        if 'condition' in name:
                            parameters += [param]
                            names += [name]
                        else:
                            param.requires_grad = False
                else:
                    parameters, names = None, None
            else:
                parameters, names = None, None


            # datasets
            _, train_dataloader, train_sampler = get_dataset(self.cfg.dataset.train, self.use_ddp)
            if self.is_main_proc:
                _, val_dataloader, _ = get_dataset(self.cfg.dataset.val, ddp=False)

            # optimizer and scheduler
            optimizer = get_optimizer(self.cfg.optimizer, model, parameters, names)
            scheduler = get_scheduler(self.cfg.scheduler, optimizer)

            # run training
            best_acc = -1
            for epoch in range(self.cfg.training.epochs):
                log_dict = {'param/lr': optimizer.state_dict()['param_groups'][0]['lr']}
                if self.use_ddp:
                    train_sampler.set_epoch(epoch)
                
                results = self.train(model, optimizer, train_dataloader)

                if self.is_main_proc:  # evaluate on one to ensure continuous mode is correct
                    if epoch % self.cfg.settings.val_every == 0 or epoch == self.cfg.training.epochs - 1:
                        if self.cfg.settings.val_single:
                            val_results = self.test(model, val_dataloader)
                            val_acc = val_results[1]
                        else:
                            val_results = [0, 0, {}, {}]
                        if self.cfg.settings.val_multi:
                            val_cont_results = self.test(model, val_dataloader, continuous=True)
                            val_acc = val_cont_results[1]
                        else:
                            val_cont_results = [0, 0, {}, {}]

                        self.log(epoch, log_dict, *results, *val_results, *val_cont_results)
                        best_acc = self.save(model, val_acc, best_acc)

                if scheduler is not None:
                    scheduler.step(epoch+1)
        else:
            self.load(model, path=self.cfg.settings.test_path, pop_condition=self.cfg.dataset.next_frames == 0)
            # run evaluation
            _, test_dataloader, _ = get_dataset(self.cfg.dataset.test, ddp=False)
            if self.cfg.settings.test_single:
                test_results = self.test(model, test_dataloader)
            if self.cfg.settings.test_multi:
                test_cont_results = self.test(model, test_dataloader, continuous=True)
            # log results
            log_str = f'{Fore.WHITE}Evaluation results:\n'
            log_dict = {}
            if self.cfg.settings.test_single:
                log_str += f'{Fore.MAGENTA} --- single step:'
                for k, v in test_results[3].items():
                    log_str += f' {k.split("_")[1]}={v*100:.1f}'
                    log_dict[f'test/{k}'] = v
                log_str += '\n'
            if self.cfg.settings.test_multi:
                log_str += f'{Fore.GREEN} --- full sequence:'
                log_str_last = f'{Fore.CYAN} --- last frame:'
                for k, v in test_cont_results[3].items():
                    v_str = f' {k.split("_")[1]}={v*100:.1f}'
                    if k.endswith('_last'):
                        log_str_last += v_str
                    else:
                        log_str += v_str
                    log_dict[f'test-cont/{k}'] = v
                log_str += '\n' + log_str_last + '\n'
            print(log_str + Style.RESET_ALL)            
            if self.wandb is not None:
                self.wandb.log(log_dict)


@hydra.main(
    version_base=None,
    config_path='../net/config', 
    config_name='dyn')
def main(cfg):
    workspace = PredictionWorkspace(cfg)
    workspace.run()


if  __name__== '__main__':
    main()
