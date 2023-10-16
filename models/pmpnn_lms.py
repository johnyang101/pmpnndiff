import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import pmpnn_utils as pu
import pmpnn_models
from models.utils import load_sd_for_matched_keys

class Generic_LM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return optimizer #{'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val/loss'}

    def load_ft_dict(self, model):
        weights_dict = torch.load(self.cfg.model.ft_path, map_location=self.device)
        if 'state_dict' in weights_dict:
            sd_pyl = weights_dict['state_dict']
            sd = {k.replace('model.', ''): v for k, v in sd_pyl.items() if 'model.' in k}
        elif 'model_state_dict' in weights_dict:
            sd = weights_dict['model_state_dict']
        else:
            raise ValueError('Could not find state_dict or model_state_dict in weights_dict.')
        
        try:
            model.load_state_dict(sd)
            print('Loaded full model weights')
        except Exception as e:
            print(e)
            model = load_sd_for_matched_keys(model, sd, self.cfg.model.ignore_keys)
        return model

class Generic_PMPNN_LM(Generic_LM):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def ce_loss_fn(self, logits, S, mask, chain_M):
        log_probs = F.log_softmax(logits, dim=-1)
        mask_for_loss = mask * chain_M
        loss, loss_av = pu.loss_smoothed(S, log_probs, mask_for_loss)
        return loss_av

    def ppl_acc(self, S, logits, mask, chain_M):
        mask_for_loss = mask * chain_M
        log_probs = F.log_softmax(logits, dim=-1)
        loss, loss_av, true_false = pu.loss_nll(S, log_probs, mask_for_loss)
        sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
        acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        weights = torch.sum(mask_for_loss).cpu().data.numpy()
        return np.exp(sum / weights), acc / weights

    def stage_log(self, stage, loss, ce_loss, ppl, acc, batch_size, commit_loss=None):
        self.log(f'{stage}/loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log(f'{stage}/ce_loss', ce_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log(f'{stage}/ppl', ppl, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log(f'{stage}/acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        if commit_loss is not None:
            self.log(f'{stage}/commit_loss', commit_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

class PMPNN_Baseline_CPD_LM(Generic_PMPNN_LM):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = pmpnn_models.PMPNN_Baseline_CPD(**self.cfg.model)
        if self.cfg.model.ft:
            self.model = self.load_ft_dict(self.model)
        self.sampling_temp = self.cfg.sampling_temp if hasattr(self.cfg, 'sampling_temp') else 0.1 #TODO: remove this hacky fix, backwards compat for old configs.
        self.save_hyperparameters()
    
    def forward(self, batch):
        X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = pu.featurize(batch, self.device)
        return self.model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
    
    def _shared_eval_step(self, batch, batch_idx):
        X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = pu.featurize(batch, self.device)
        B, N = S.shape
        
        logits = self.model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        ce_loss = self.ce_loss_fn(logits, S, mask, chain_M)
        loss = ce_loss
        
        ppl, acc = self.ppl_acc(S, logits, mask, chain_M)
        
        return loss, ce_loss, ppl, acc, B
    
    def training_step(self, batch, batch_idx):
        loss, ce_loss, ppl, acc, B = self._shared_eval_step(batch, batch_idx)
        
        self.stage_log('train', loss, ce_loss, ppl, acc, B)
        return loss
    
    def validation_step(self, batch, batch_idx): 
        loss, ce_loss, ppl, acc, B = self._shared_eval_step(batch, batch_idx)
        
        self.stage_log('val', loss, ce_loss, ppl, acc, B)
        
        return ppl, acc 
    