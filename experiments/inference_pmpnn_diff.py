from __future__ import print_function
import pandas as pd
import os, json
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
from typing import List

import pytorch_lightning as pl
import hydra

from models.diffusion_lms import UNL_Absorbing_Diff_LM
from data.data_objs import PMPNNBatch
import experiments.utils as eu


def sampling_fn(lm, obj, sampling_type, **kwargs):
    if sampling_type == 'sample':
        return lm.sample(obj)
    elif sampling_type == 'purity_sample':
        return lm.purity_sample(obj, **kwargs)
    else:
        raise ValueError(f'Invalid sampling_type: {sampling_type}')

def sample_metric(lm, jsonl_file, name, sampling_type, device='cpu', **kwargs):

    output_df = pd.DataFrame(columns=['header', 'sequence', 'acc'])

    with open(jsonl_file) as f:
        lines = f.readlines()
        for _ in range(8):
            for line in tqdm(lines):
                entry = json.loads(line)
                header = entry['name']
                batch = [entry]
                obj = PMPNNBatch(batch, device)
                obj, acc = sampling_fn(lm, obj, sampling_type, **kwargs)
                seq = eu.format_sequences(obj.x_t, mask=obj.mask, lengths=obj.lengths, names=[b['name'] for b in obj.batch])
                if isinstance(seq, list) and len(seq) == 1:
                    seq = seq[0]
                output_df = output_df.append({'header': header, 'sequence': seq, 'acc': acc.item()}, ignore_index=True)
            
    '''Save `output_df` with a context manager'''
    if not os.path.exists('./sampling_results'):
        os.makedirs('./sampling_results')
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    params = f"sampling_type={sampling_type}_params={kwargs}"
    output_path = f"./sampling_results/sampling_{timestamp}_{name}_{params}.csv"
    
    with open(output_path, 'w') as f:
        output_df.to_csv(f)
        
@hydra.main(config_path="../config/clean", config_name="base")
def main(cfg):
    lm = UNL_Absorbing_Diff_LM.load_from_checkpoint(cfg=cfg.lm, checkpoint_path=cfg.experiment.ckpt_path, map_location='cpu')
    device, replica_id = eu.initialize_device_and_model(lm)
    sample_metric(lm, cfg.dm.test_jsonl.path, cfg.experiment.name, cfg.sampling_type, device=device, **cfg.sampling_kwargs)