from __future__ import print_function
import numpy as np
import pandas as pd
import os, json, math, copy, argparse
from tqdm import tqdm
from datetime import datetime
import functools as fn

import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.distributions as dists
from torchtyping import TensorType
from typing import List

import pytorch_lightning as pl
from omegaconf import OmegaConf

from data.data_objs import *
from models.pmpnn_models import PMPNN_Baseline_Diff
import experiments.utils as eu

DEBUG = False
NUM_TIMESTEPS = 100
NUM_CLASSES = 21
MAX_PROTEIN_LENGTH = 10000

import hydra
with hydra.initialize('../config/pyl/'):
    conf = hydra.compose(config_name='PMPNN_UNL_exp.yaml')

def create_model_config(train=True, pmpnn_ft_path='../weights/pmpnn_arm.pt'):
    augment_eps = 0.02 if train else 0.00
    return {
        'num_letters': NUM_CLASSES,
        'node_features': 128,
        'edge_features': 128,
        'hidden_dim': 128,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'vocab': NUM_CLASSES,
        'k_neighbors': 48,
        'augment_eps': augment_eps,
        'dropout': 0.1,
        'ft': True,
        'ft_path': pmpnn_ft_path,
        'disable_random_decoding_order': False,
        'name': 'PMPNN_Baseline_Diff',
        'absorbing': True,
        'num_classes': NUM_CLASSES,
        'embedding_cfg': {
            'max_len': MAX_PROTEIN_LENGTH,
            'T': NUM_TIMESTEPS,
            'input_size': NUM_CLASSES,
            'pos_embed_size': 64,
            'output_embed_size': 128
        }
    }
    

test_model_conf = create_model_config(train=False)
dev_denoise_model = PMPNN_Baseline_Diff(**test_model_conf).train()

# Usage
loader = eu.ModelLoader()
dev_denoise_model = loader.load_weights(dev_denoise_model, test_model_conf['ft_path'])

# %%
def to_one_hot(indices, num_classes):
    """
    Convert a 2D index tensor (batched) into one-hot format.
    """
    B, N = indices.size()
    one_hot = torch.zeros(B, N, num_classes, dtype=torch.float32, device=indices.device)
    return one_hot.scatter_(2, indices.unsqueeze(-1), 1.0)

# %%
class UNL_Absorbing_Diff_LM(pl.LightningModule):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.debug = debug

        self.shape = cfg.shape #tuple(cfg.latent_shape)
        self.num_timesteps = cfg.num_timesteps if not self.debug else 10
        
        self.num_classes = cfg.num_classes #Expects alphabet size + 1 for mask
        self.mask_id = self.num_classes - 1
        self.n_samples = cfg.n_samples
        self.loss_type = cfg.loss_type
        print(f'~~~~ LOSS TYPE {self.loss_type} ~~~~~')
        self.mask_schedule = cfg.mask_schedule
        self.aux_weight = cfg.aux_weight
        
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))
        
        self._denoise_fn = dev_denoise_model

        assert self.mask_schedule in ['random', 'fixed']
        
        self.temp = cfg.temp

    def freeze_pmpnn_params(self, params):
        for name, param in self._denoise_fn.named_parameters():
            if name in params:
                param.requires_grad = False
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def sample_time(self, b, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t: TensorType['B'] = torch.randint(0, self.num_timesteps, (b,)).long()
            pt: TensorType['B'] = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError
    
    def get_obj(self, batch):
        return PMPNNBatch(batch, self.device)

    def q_sample(self, x_0, t):
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.rand_like(x_t.float()) < (t.float().unsqueeze(-1) / self.num_timesteps)
        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask
    
    def q_sample_mlm(self, x_0, t):
        # samples q(x_t | x_0)
        # fixed noise schedule, masks exactly int(t/T * latent_size) tokens
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.zeros_like(x_t).to(torch.bool)

        # TODO: offset so each n_masked_tokens is picked with equal probability
        n_masked_tokens = (t.float() / self.num_timesteps) * x_t.size(1)
        n_masked_tokens = torch.round(n_masked_tokens).to(torch.int64)
        n_masked_tokens[n_masked_tokens == 0] = 1
        ones = torch.ones_like(mask[0]).to(torch.bool).to(x_0.device)

        for idx, n_tokens_to_mask in enumerate(n_masked_tokens):
            index = torch.randperm(x_0.size(1))[:n_tokens_to_mask].to(x_0.device)
            mask[idx].scatter_(dim=0, index=index, src=ones)

        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask
            
    def training_step(self, batch, batch_idx):
        obj = self.get_obj(batch)

        # make x noisy and denoise
        assert obj.x_start is not None, 'x_start is None. Check the get_obj method.'
        '''Assert obj.x_start is a torch tensor of shape [B, N].'''
        assert obj.x_start.shape == (obj.B, obj.N), 'Expected shape [B, N].'
        
        obj.to(self.device)
        
        if len(obj.x_start.shape) == 1:
            obj.x_start = obj.x_start.unsqueeze(0)

        assert len(obj.x_start.shape) == 2, 'Expected shape [B, N].' 
        
        x_0 = obj.x_start
        
        t, pt = self.sample_time(obj.B, method='importance') #Size is [B,]
        t, pt = t.to(self.device), pt.to(self.device)
        
        if self.mask_schedule == 'random':
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)
        else:
            raise ValueError('Unknown mask schedule: {}'.format(self.mask_schedule))

        obj.x_t = x_t
        
        x_0_hat_logits = self._denoise_fn(obj, t=t).permute(0, 2, 1) #TODO:
        
        cross_entropy_loss: TensorType['B'] = (F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none') * obj.mask).sum(1)
        t_denom = torch.where(t == 0, torch.ones_like(t), t)
        pt_denom = torch.where(pt == 0, torch.ones_like(pt), pt)
        vb_loss = cross_entropy_loss / t_denom
        vb_loss = vb_loss / pt_denom
        vb_loss = vb_loss / (math.log(2) * obj.mask.sum(1))
        if self.loss_type == 'elbo':
            loss = vb_loss
        elif self.loss_type == 'mlm':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1  # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif self.loss_type == 'reweighted_elbo':
            weight = (1 - (t / self.num_timesteps))
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * obj.mask.sum(1))
        else:
            raise ValueError

        # Track loss at each time step history for bar plot
        Lt2_prev = self.loss_history.gather(dim=0, index=t)
        new_loss_history = (0.1 * loss + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)

        self.loss_history.scatter_(dim=0, index=t, src=new_loss_history)

        # Track loss at each time step for importance sampling
        Lt2 = vb_loss.detach().clone().pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2).to(self.loss_history.dtype))

        mean_vb_loss = vb_loss.mean()
        self.log('train/vb_loss', mean_vb_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=obj.B)
        self.log('train/loss', loss.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=obj.B)
        assert torch.isnan(loss).sum() == 0, 'Loss is NaN'
        
        return loss.mean()

def initialize_device_and_model(lm):
    replica_id = int(eu.get_free_gpu()) #TODO: 
    device = f'cuda:{replica_id}'
    lm.to(device)
    return device, replica_id

def trad_sample(lm, obj, stride=1):
    print('Sampling trad')
    b = obj.B
    device = lm.device
    obj.x_t = torch.ones_like(obj.x_start, device=lm.device).long() * lm.mask_id
    unmasked = ~obj.mask.bool()
    sample_steps = list(range(1, lm.cfg.num_timesteps, stride)) #NOTE: We don't add +1 bc of TimestepEmbedding bugs.

    X, S, mask, chain_M, residue_idx, chain_encoding_all = obj.X, obj.S, obj.mask, obj.chain_M, obj.residue_idx, obj.chain_encoding_all
    h_V, h_E, E_idx = lm._denoise_fn.encode(X, S, mask, chain_M, residue_idx, chain_encoding_all)

    for t in reversed(sample_steps):
        print(f'Sample timestep {t:4d}', end='\r')
        t = torch.full((b,), t, device=device, dtype=torch.long)

        # where to unmask
        changes = torch.rand(obj.x_t.shape, device=device) < 1/t.float().unsqueeze(-1)
        # don't unmask somewhere already unmasked
        changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
        # update mask with changes
        unmasked = torch.bitwise_or(unmasked, changes)

        x_t = obj.x_t
        one_hot_x_t = F.one_hot(x_t, lm.num_classes)
        h_S = lm._denoise_fn.W_s(obj.B, t, obj.N, one_hot_x_t)

        x_0_logits = lm._denoise_fn.decode(mask, chain_M, h_V, h_E, E_idx, h_S) 
        # scale by temperature
        x_0_logits = x_0_logits / lm.cfg.temp
        x_0_dist = dists.Categorical(
            logits=x_0_logits)
        x_0_hat = x_0_dist.sample().long()
        obj.x_t[changes] = x_0_hat[changes]
    assert (torch.where(obj.x_t == lm.mask_id, torch.ones_like(obj.x_t), torch.zeros_like(obj.x_t)) * obj.mask).sum() == 0, 'Mask ID still in x_t'
    acc = torch.sum(torch.eq(obj.x_t[obj.mask.bool()], obj.x_start[obj.mask.bool()])) / obj.mask.sum()
    return obj, acc

def purity_sample(lm, obj, stride=1, max_sample_ratio_per_step=0.05, prior_weight=0.5):
    b = obj.B
    device = lm.device
    obj.x_t = torch.ones_like(obj.x_start, device=lm.device).long() * lm.mask_id
    unmasked = ~obj.mask.bool()
    sample_steps = list(range(1, lm.num_timesteps, stride)) #NOTE: We don't add +1 bc of TimestepEmbedding bugs.
    X, S, mask, chain_M, residue_idx, chain_encoding_all = obj.X, obj.S, obj.mask, obj.chain_M, obj.residue_idx, obj.chain_encoding_all
    
    h_V, h_E, E_idx = lm._denoise_fn.encode(X, S, mask, chain_M, residue_idx, chain_encoding_all)
    
    max_sample_per_step = round((obj.N * max_sample_ratio_per_step))
    
    sampled = torch.zeros(obj.B)
    to_sample = obj.N
    
    for t in reversed(sample_steps):
        print(f'Sample timestep {t:4d}', end='\r')
        t = torch.full((b,), t, device=device, dtype=torch.long)
        
        x_t = log_x_idx = obj.x_t
        one_hot_x_t: TensorType[obj.B, obj.N, lm.num_classes] = F.one_hot(x_t, lm.num_classes)
        
        #Embed timestep in sequence.
        h_S = lm._denoise_fn.W_s(obj.B, t, obj.N, one_hot_x_t)

        x_0_logits = lm._denoise_fn.decode(mask, chain_M, h_V, h_E, E_idx, h_S).permute(0, 2, 1)  

        # x_0_logits = lm._denoise_fn(obj, t=t)
        log_x_recon = x_0_logits

        score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
        score /= (score.max(dim=1, keepdim=True).values + 1e-10)

        # probability adjust parameter, prior_weight: 'r' in Equation.11 of Improved VQ-Diffusion
        prob = ((1 + score * prior_weight).unsqueeze(1) * log_x_recon).softmax(dim=1)
        prob = prob.log().clamp(-70, 0)

        x_0_logits = prob / lm.cfg.temp
        x_0_dist = dists.Categorical(
            logits=x_0_logits.permute(0, 2, 1))
        x_0_hat = out_idx = x_0_dist.sample().long()

        out2_idx = log_x_idx.clone()
        _score = score.clone()
        if _score.sum() < 1e-6:
            _score += 1
        _score[log_x_idx != lm.mask_id] = 0

        for i in range(obj.B):
            n_sample = min(round((to_sample - sampled[i].item())), max_sample_per_step)
            if to_sample - sampled[i] - n_sample == 1:
                n_sample = round(to_sample - sampled[i].item())
            if n_sample <= 0:
                continue
            # to_sample -= n_sample

            sel = changes = torch.multinomial(_score[i], n_sample)
            
            out2_idx[i][sel] = out_idx[i][sel]
            sampled[i] += ((out2_idx[i] != lm.mask_id).sum() - (log_x_idx[i] != lm.mask_id).sum()).item()
            obj.x_t[i] = out2_idx[i]
            
    assert (torch.where(obj.x_t == lm.mask_id, torch.ones_like(obj.x_t), torch.zeros_like(obj.x_t)) * obj.mask).sum() == 0, 'Mask ID still in x_t'
    acc = torch.sum(torch.eq(obj.x_t[obj.mask.bool()], obj.x_start[obj.mask.bool()])) / obj.mask.sum()
    return obj, acc


def run(lm, name, sampling_type, stride=1, max_sample_ratio_per_step=0.05, prior_weight=2, device='cpu'):

    output_df = pd.DataFrame(columns=['header', 'sequence', 'acc'])

    sampling_fn = fn.partial(purity_sample, max_sample_ratio_per_step=max_sample_ratio_per_step, prior_weight=prior_weight) if sampling_type == 'purity_sample' else trad_sample

    jsonl_file = conf.dm.test_jsonl_path
    with open(jsonl_file) as f:
        lines = f.readlines()
        for _ in range(8):
            for line in tqdm(lines):
                entry = json.loads(line)
                header = entry['name']
                batch = [entry]
                obj = PMPNNBatch(batch, device)
                obj, acc = sampling_fn(lm, obj, stride=stride)
                seq = eu.format_sequences(obj.x_t, mask=obj.mask, lengths=obj.lengths, names=[b['name'] for b in obj.batch])
                if isinstance(seq, list) and len(seq) == 1:
                    seq = seq[0]
                output_df = output_df.append({'header': header, 'sequence': seq, 'acc': acc.item()}, ignore_index=True)
            
    '''Save `output_df` with a context manager'''
    if not os.path.exists('./sampling_results'):
        os.makedirs('./sampling_results')
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if sampling_type == 'purity_sample':
        params = f"sampling_type={sampling_type}_stride={stride}_max_sample_ratio_per_step={max_sample_ratio_per_step}_prior_weight={prior_weight}"
    elif sampling_type == 'sample':
        params = f"sampling_type={sampling_type}_stride={stride}"
    else:
        params = f"sampling_type={sampling_type}"
        
    output_path = f"./sampling_results/sampling_{timestamp}_{name}_{params}.csv"
    
    with open(output_path, 'w') as f:
        output_df.to_csv(f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_type", help="The type of sampling to use.")
    parser.add_argument("--stride", type=int, default=1, help="The stride for the sampling.")
    parser.add_argument("--max_sample_ratio_per_step", type=float, default=0.05, help="The maximum sample ratio per step.")
    parser.add_argument("--prior_weight", type=float, default=2.0, help="The weight of the prior.")
    parser.add_argument("--ckpt_path", type=str, default='../weights/pmpnn_diff.ckpt', help="Diff LM ckpt path")
    parser.add_argument("--device", type=int, default=None, help="Device to run on.")
    parser.add_argument("--name", type=str, default=None, help="Name of the experiment.")
    
    args = parser.parse_args()
    
    lm_conf = OmegaConf.create({
        'shape': None,
        'n_samples': 64,
        'num_timesteps': 100,
        'loss_type': 'reweighted_elbo', #NOTE: CHANGED TO REWEIGHTED ELBO
        'mask_schedule': 'random',
        'aux_weight': 0.01,
        'temp': 1.0,
        'token': {'type': 'prot_seq'},
        'learning_rate': 0.0001,
        'ignore_keys': [],
        'sampling_temp': 0.1,
        'freeze_ae': True,
        'lm_name': 'UNL_Absorbing_Diff_LM',
        'model_type': 'PMPNN',
        'num_classes': 21,
        'save_name': 'no_causal_attn',
        }
    )
    lm = UNL_Absorbing_Diff_LM.load_from_checkpoint(cfg=lm_conf, checkpoint_path=args.ckpt_path, map_location='cpu')
    lm.freeze_pmpnn_params(loader.loaded_params)
    device, replica_id = initialize_device_and_model(lm)

    run(lm, args.name, args.sampling_type, args.stride, args.max_sample_ratio_per_step, args.prior_weight, device=device)