import numpy as np
import typing as T
from typing import List
import torch
import os, copy, GPUtil
from datetime import datetime

def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened

class ModelLoader:

    def __init__(self):
        self.loaded_params = []

    def load_matching_keys(self, model, sd, ignore_keys: List[str] = []):
        """Load model parameters for matching keys between model and given state_dict."""
        model_sd = copy.deepcopy(model.state_dict())
        for param in model_sd.keys():
            if param in sd and param not in ignore_keys:
                try:
                    print(f'Loading {param}')
                    model_sd[param] = sd[param]
                    self.loaded_params.append(param)
                except Exception as e:
                    print(e, f'\nFailed to load {param}')
        model.load_state_dict(model_sd)
        return model

    def load_weights(self, model, weights_path: str):
        """Attempt to load weights into the given model."""
        weights_dict = torch.load(weights_path, map_location='cpu')
        sd = self._extract_state_dict(weights_dict)
        
        try:
            model.load_state_dict(sd)
            print('Loaded full model weights')
        except Exception as e:
            print(e)
            ignore_keys = self._generate_ignore_keys(sd)
            model = self.load_matching_keys(model, sd, ignore_keys)
        return model

    def _extract_state_dict(self, weights_dict):
        """Extract state_dict from given weights dictionary."""
        if 'state_dict' in weights_dict:
            sd_pyl = weights_dict['state_dict']
            return {k.replace('model.', ''): v for k, v in sd_pyl.items() if 'model.' in k}
        elif 'model_state_dict' in weights_dict:
            return weights_dict['model_state_dict']
        else:
            raise ValueError('Could not find state_dict or model_state_dict in weights_dict.')

    def _generate_ignore_keys(self, sd, keyword='decoder'):
        """Generate list of keys to be ignored during loading."""
        keyword_keys = [k for k in sd.keys() if keyword in k]
        return ['features.edge_embedding.weight', 'W_out.weight', 'W_out.bias'] + keyword_keys

def get_free_gpu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # Set environment variables for which GPUs to use.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    chosen_gpu = ''.join(
        [str(x) for x in GPUtil.getAvailable(order='memory')])
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen_gpu
    print(f"Using GPUs: {chosen_gpu}")
    return chosen_gpu

'''Given a tensor of indices and an alphabet, return the string of characters'''
def indices_to_string(indices, alphabet, mask, length, pad_token='<mask>'):
    assert len(indices.shape) == 1, 'indices must be 1D'
    assert indices.shape == mask.shape, 'indices and mask must have same shape'
    x = ''.join([alphabet[int(res_ind.item())] if mask[iter_ind] == 1 else pad_token for iter_ind, res_ind in enumerate(indices) if iter_ind < length])
    return x

def format_sequences(seq_ind, mask, lengths: np.ndarray, names: T.List[str]) -> T.List[str]:
    '''
    seq_ind: (batch_size, seq_len)
    mask: (batch_size, seq_len)
    lengths: list of sequence lengths
    '''
    
    assert seq_ind.shape == mask.shape and seq_ind.shape[1] == lengths.max(), 'seq_ind, mask, and lengths must have same shape'
    assert len(names) == seq_ind.shape[0], 'names must have same length as batch_size'
    formatted_sequences = [indices_to_string(seq_ind[i], 'ACDEFGHIKLMNPQRSTVWYX', mask[i], lengths[i], pad_token='X') for i in range(seq_ind.shape[0])]
    return formatted_sequences

def header_sequence_tuples_from_objs(objs):
    return [x for o in objs for x in zip([b['name'] for b in o.batch], format_sequences(o.x_t, mask=o.mask, lengths=o.lengths, names=[b['name'] for b in o.batch]))]

def save_results(output_df, sampling_type, stride, max_sample_ratio_per_step, prior_weight):
    if not os.path.exists('./sampling_results'):
        os.makedirs('./sampling_results')
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    params = get_params_string(sampling_type, stride, max_sample_ratio_per_step, prior_weight)
    
    output_path = f"./sampling_results/sampling_{timestamp}_{params}.csv"
    with open(output_path, 'w') as f:
        output_df.to_csv(f)

def get_params_string(sampling_type, stride, max_sample_ratio_per_step, prior_weight):
    if sampling_type == 'purity_sample':
        params = f"sampling_type={sampling_type}_stride={stride}_max_sample_ratio_per_step={max_sample_ratio_per_step}_prior_weight={prior_weight}"
    elif sampling_type == 'sample':
        params = f"sampling_type={sampling_type}_stride={stride}"
    else:
        params = f"sampling_type={sampling_type}"
    return params

def initialize_device_and_model(lm):
    replica_id = int(get_free_gpu())
    device = f'cuda:{replica_id}'
    lm.to(device)
    return device, replica_id
