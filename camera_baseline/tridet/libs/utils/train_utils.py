import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pandas as pd

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm


################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
        optimizer,
        optimizer_config,
        num_iters_per_epoch,
        last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # get eta min
        eta_min = optimizer_config["eta_min"]

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                eta_min=eta_min,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get eta min
        eta_min = optimizer_config["eta_min"]

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                eta_min=eta_min,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch(train_loader, model, optimizer, scheduler, model_ema=None, clip_grad_l2norm=-1):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses = model(video_list)
        losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad_l2norm)
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # measure elapsed time (sync all kernels)
        torch.cuda.synchronize()
        batch_time.update((time.time() - start))
        start = time.time()

        # track all losses
        for key, value in losses.items():
            # init meter if necessary
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            # update
            losses_tracker[key].update(value.item())

    return losses_tracker['final_loss'].avg


def valid_one_epoch(val_loader, model):
    """Test the model on the validation set"""
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }


    
    # loop over validation set
    for _, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)
            # upack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    return 0.0, results
    
    '''
    
    
    c_1= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-25_16-36-26\\unprocessed_results\\v_seg_wear_split_1.csv')
    c_2= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-25_16-36-26\\unprocessed_results\\v_seg_wear_split_2.csv')
    c_3= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-25_16-36-26\\unprocessed_results\\v_seg_wear_split_3.csv')
    i_1= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-25_15-05-21\\unprocessed_results\\v_seg_wear_split_1.csv')
    i_2= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-25_15-05-21\\unprocessed_results\\v_seg_wear_split_2.csv')
    i_3= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-25_15-05-21\\unprocessed_results\\v_seg_wear_split_3.csv')

    
    merged_df1 = pd.concat([c_1, i_1], ignore_index=True)
    merged_df2 = pd.concat([c_2, i_2], ignore_index=True)
    merged_df3 = pd.concat([c_3, i_3], ignore_index=True)

    sorted_df1 = merged_df1.sort_values(['video-id', 'score'], ascending=[True, False])
    top_2000_df1 = sorted_df1.groupby('video-id').head(2000)
    
    sorted_df2 = merged_df2.sort_values(['video-id', 'score'], ascending=[True, False])
    top_2000_df2 = sorted_df2.groupby('video-id').head(2000)

    sorted_df3 = merged_df3.sort_values(['video-id', 'score'], ascending=[True, False])
    top_2000_df3 = sorted_df3.groupby('video-id').head(2000)

    if(i==0):
        df = top_2000_df1
    elif(i==1):
        df = top_2000_df2
    elif(i==2):
        df = top_2000_df3

    video_ids = df['video-id'].values
    t_starts = df['t-start'].values
    t_ends = df['t-end'].values
    labels = df['label'].values
    scores = df['score'].values

    # Add the values to the results dictionary
    results['video-id'].extend(video_ids)
    results['t-start'].extend(t_starts)
    results['t-end'].extend(t_ends)
    results['label'].extend(labels)
    results['score'].extend(scores)

    # Convert the lists to numpy arrays
    results['video-id'] = np.array(results['video-id'])
    results['t-start'] = np.array(results['t-start'])
    results['t-end'] = np.array(results['t-end'])
    results['label'] = np.array(results['label'])
    results['score'] = np.array(results['score'])

    return 0.0, results

    
    csv_file1 = 'D:\\wear_main\\logs\\actionformer\\2023-07-10_21-26-44\\unprocessed_results\\v_seg_wear_split_1.csv'
    csv_file2 = 'D:\\wear_main\\logs\\actionformer\\2023-07-10_21-26-44\\unprocessed_results\\v_seg_wear_split_2.csv'
    csv_file3 = 'D:\\wear_main\\logs\\actionformer\\2023-07-10_21-26-44\\unprocessed_results\\v_seg_wear_split_3.csv'
    

    if(i==0):
        df = pd.read_csv('D:\\wear_main\\logs\\actionformer\\2023-07-10_21-26-44\\unprocessed_results\\v_seg_wear_split_1.csv')
    elif(i==1):
        df = pd.read_csv('D:\\wear_main\\logs\\actionformer\\2023-07-10_21-26-44\\unprocessed_results\\v_seg_wear_split_2.csv')
    elif(i==2):
        df = pd.read_csv('D:\\wear_main\\logs\\actionformer\\2023-07-10_21-26-44\\unprocessed_results\\v_seg_wear_split_3.csv')


    # Extract the values from the DataFrame
    video_ids = df['video-id'].values
    t_starts = df['t-start'].values
    t_ends = df['t-end'].values
    labels = df['label'].values
    scores = df['score'].values

    # Add the values to the results dictionary
    results['video-id'].extend(video_ids)
    results['t-start'].extend(t_starts)
    results['t-end'].extend(t_ends)
    results['label'].extend(labels)
    results['score'].extend(scores)

    # Convert the lists to numpy arrays
    results['video-id'] = np.array(results['video-id'])
    results['t-start'] = np.array(results['t-start'])
    results['t-end'] = np.array(results['t-end'])
    results['label'] = np.array(results['label'])
    results['score'] = np.array(results['score'])

    return 0.0, results
    '''