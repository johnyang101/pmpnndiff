from __future__ import print_function
import numpy as np
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
from models.pmpnn import PMPNN_Baseline_CPD
from data.data_objs import PMPNNBatch
import experiments.utils as eu

def pmpnn_sample(lm, obj, device):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    chain_M_pos = obj.chain_M
    omit_AAs_np = np.array([AA in '-' for AA in alphabet]).astype(np.float32)
    bias_AAs_np = np.zeros(len(alphabet))
    bias_by_res_all = torch.zeros((obj.S.shape[0], obj.S.shape[1], len(alphabet)), device=device)
    
    decoding_order = torch.randn(obj.chain_M.shape, device=device)
    
    output_dict = lm.sample(obj.X, decoding_order, obj.S, obj.chain_M, obj.chain_encoding_all, obj.residue_idx, mask=obj.mask, chain_M_pos=chain_M_pos, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, bias_by_res=bias_by_res_all)
    
    obj.x_t = output_dict['S']
    accuracy = obj.calculate_accuracy()
    return obj, accuracy

def sampling_fn(lm, obj, sampling_type, device='cpu', **kwargs):
    if sampling_type == 'sample':
        return lm.sample(obj)
    elif sampling_type == 'purity_sample':
        return lm.purity_sample(obj, **kwargs)
    elif sampling_type == 'pmpnn':
        return pmpnn_sample(lm, obj, device)
    else:
        raise ValueError(f'Invalid sampling_type: {sampling_type}')

def sample_metric(lm, jsonl_file, name, sampling_type, device='cpu', **kwargs):

    rows = []

    with open(jsonl_file) as f:
        lines = f.readlines()
        for _ in range(8):
            for line in tqdm(lines):
                entry = json.loads(line)
                header = entry['name']
                batch = [entry]
                obj = PMPNNBatch(batch, device)
                obj, acc = sampling_fn(lm, obj, sampling_type, device=device, **kwargs)
                seq = obj.sampled_seq_string()
                if isinstance(seq, list) and len(seq) == 1:
                    seq = seq[0]

                # Append dictionary to rows list
                rows.append({'header': header, 'sequence': seq, 'acc': acc})

    # Create DataFrame from rows list
    output_df = pd.DataFrame(rows, columns=['header', 'sequence', 'acc'])
            
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
    strat = cfg.sampling_type
    if strat == 'sample' or strat == 'purity_sample':
        model = UNL_Absorbing_Diff_LM.load_from_checkpoint(cfg=cfg.lm, checkpoint_path=cfg.experiment.ckpt_path, map_location='cpu')
    elif strat == 'pmpnn':
        model = PMPNN_Baseline_CPD(**cfg.lm.model)
        checkpoint = torch.load(cfg.experiment.ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f'Invalid sampling_type: {strat}')
    
    device, replica_id = eu.initialize_device_and_model(model)
    sample_metric(model, cfg.dm.test_jsonl_path, cfg.experiment.name, cfg.sampling_type, device=device, **cfg.sampling_kwargs)
    
if __name__ == '__main__':
    main()