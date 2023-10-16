import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import os, GPUtil
from typing import Dict, List, Tuple, Union

import hydra, logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import data.datamodules as dm
from models.pmpnn_lms import PMPNN_Baseline_CPD_LM
from models.diffusion_lms import UNL_Absorbing_Diff_LM

import experiments.utils as eu

class Experiment:
    def __init__(self,
            *,
            conf: DictConfig):
        """Initialize experiment.
        Args:
            conf: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._conf = conf
        self._exp_conf = conf.experiment
            
        self._lm_conf = conf.lm
        self._dm_conf = conf.dm
        self._trainer_conf = conf.trainer
        self._callbacks_conf = conf.callbacks
        
        self._use_wandb = self._exp_conf.use_wandb

        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (
                f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        
        self.debug = self._conf.debug
        if self.debug:
            os.environ["WANDB_MODE"] = "dryrun"
            self._exp_conf.name = f'{self._exp_conf.name}_debug'
            if self._conf.debug_lnc: #Logging and Checkpointing
                self._trainer_conf.check_val_every_n_epoch = 1
                self._trainer_conf.log_every_n_steps = 1
                self._trainer_conf.max_epochs = 2
                self._trainer_conf.overfit_batches = 2
                self._trainer_conf.limit_train_batches = 2
                self._trainer_conf.limit_val_batches = 2
                self._trainer_conf.limit_predict_batches = 2
            else:
                self._trainer_conf.fast_dev_run = 1
                if self._exp_conf.mode == 'train':
                    self._use_wandb = False
        
        self._lm = self._init_lm(self._lm_conf)
        self._dm = self._init_dm(self._dm_conf)
    
    def _init_lm(self, lm_conf: DictConfig):
        """Initialize lm.
        Args:
            lm_conf: lm configuration.
        Returns:
            lm.
        """
        # if lm_conf.lm_name == 'GVP_Paper_CPD_LM':
        #     return GVP_Paper_CPD_LM(lm_conf)
        # elif lm_conf.lm_name == 'GVP_VQ_CPD_LM':
        #     return GVP_VQ_CPD_LM(lm_conf)
        # elif lm_conf.lm_name == 'GVP_OTVQ_CPD_LM':
        #     return GVP_OTVQ_CPD_LM(lm_conf)
        # elif lm_conf.lm_name == 'Hoogeboom_Uniform_Diff_LM':
        #     return Hoogeboom_Uniform_Diff_LM(lm_conf, self.debug)
        # elif lm_conf.lm_name == 'PMPNN_OTVQ_CPD_LM':
        #     return PMPNN_OTVQ_CPD_LM(lm_conf)
        # elif lm_conf.lm_name == 'PMPNN_VQ_CPD_LM':
        #     return PMPNN_VQ_CPD_LM(lm_conf)
        if lm_conf.lm_name == 'PMPNN_Baseline_CPD_LM':
            return PMPNN_Baseline_CPD_LM(lm_conf)
        # elif lm_conf.lm_name == 'PMPNN_MaskedSeq_CPD_LM':
            # return PMPNN_MaskedSeq_CPD_LM(lm_conf)
        # elif lm_conf.lm_name == 'PMPNN_MaskedESM_CPD_LM':
            # return PMPNN_MaskedESM_CPD_LM(lm_conf)
        # elif lm_conf.lm_name == 'PMPNN_OTVQ_MSCPD_LM':
            # return PMPNN_OTVQ_MSCPD_LM(lm_conf)
        # elif lm_conf.lm_name == 'PMPNN_OTVQ_MESMCPD_LM':
            # return PMPNN_OTVQ_MESMCPD_LM(lm_conf)
        # elif lm_conf.lm_name == 'MSFT_Absorbing_Diff_LM':
            # return MSFT_Absorbing_Diff_LM(lm_conf, self.debug)
        # elif lm_conf.lm_name == 'Latent_Uniform_Diff_LM':
            # return Latent_Uniform_Diff_LM(lm_conf, self.debug)
        # elif lm_conf.lm_name == 'Latent_Absorbing_Diff_LM':
            # return Latent_Absorbing_Diff_LM(lm_conf, self.debug)
        elif lm_conf.lm_name == 'UNL_Absorbing_Diff_LM':
            return UNL_Absorbing_Diff_LM(lm_conf, self.debug)
        # elif lm_conf.lm_name == 'Latent_UNLAbs_Diff_LM':
            # return Latent_UNLAbs_Diff_LM(lm_conf, self.debug)
        else:
            raise ValueError(f'Unknown lm: {lm_conf.lm_name}')
    
    def _init_dm(self, dm_conf: DictConfig):
        """Initialize dm.
        Args:
            dm_conf: dm configuration.
        Returns:
            dm.
        """
        
        if dm_conf.name == 'PMPNN_JSONL_DM':
            return dm.PMPNN_JSONL_DM(dm_conf, self.debug, dm_conf.truncate)
        elif dm_conf.name == 'PMPNN_MC_DM':
            return dm.PMPNN_MC_DM(dm_conf, self.debug, dm_conf.truncate)
        else:
            raise ValueError(f'Unknown dm: {dm_conf.name}')
    
    def choose_gpu(self):
        available_gpus = ''.join(
            [str(x) for x in GPUtil.getAvailable(
                order='memory', limit = 8, maxLoad=0.1, maxMemory=0.1,)])
        if not available_gpus:
            raise ValueError('No GPUs available.')
        
        if torch.cuda.is_available():
            if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
                hydra_job_num = HydraConfig.get().job.num
                if hydra_job_num > len(available_gpus): #For Optuna sweeps
                    hydra_job_num %= len(available_gpus)
                chosen_gpu = available_gpus[hydra_job_num]
            else:
                chosen_gpu = available_gpus[0]
            self._log.info(f"Using GPU: {chosen_gpu}")
            chosen_gpu = int(chosen_gpu)
        else:
            chosen_gpu = 0
        
        return chosen_gpu
    
    def start_training(self, return_logs=False):
        chosen_gpu = self.choose_gpu()
        return self.train(
                chosen_gpu, 1, return_logs=return_logs, init_wandb=True)
    
    def init_wandb(self):
        if self._use_wandb:
            
            self._log.info('Initializing Wandb.')
            conf_dict = OmegaConf.to_container(self._conf, resolve=True)
            cfd = dict(eu.flatten_dict(conf_dict))
            self._wandb_logger = WandbLogger(config=cfd, **self._exp_conf.wandb_logger)
        else:
            self._wandb_logger = None
            
    def get_callbacks(self, callbacks_conf: DictConfig):
        callbacks_list = []
        if callbacks_conf.es_cb and callbacks_conf.es_cb_enabled:
            callbacks_list.append(EarlyStopping(**callbacks_conf.es_cb))
        if callbacks_conf.ckpt_cb and callbacks_conf.ckpt_cb_enabled:
            callbacks_list.append(ModelCheckpoint(**callbacks_conf.ckpt_cb))
        return callbacks_list
    
    def train(self, replica_id, num_replicas, return_logs=False, init_wandb=False):

        self.init_wandb() 
            
        if torch.cuda.is_available():
            device = f"cuda:{replica_id}"
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        
        callbacks_list = self.get_callbacks(self._callbacks_conf)
        trainer_loggers = [self._wandb_logger] if self._wandb_logger else []
        
        self.trainer = pl.Trainer(devices=[replica_id], 
                             callbacks=callbacks_list, 
                             logger=trainer_loggers, 
                             **self._trainer_conf)
            
        self.trainer.fit(self._lm, datamodule=self._dm, ckpt_path=self._exp_conf.ckpt_path)
        train_metrics_dict = self.trainer.callback_metrics
        self._log.info('Finished training.')
        
        metrics_dict = train_metrics_dict #TODO: Add test metrics
        return metrics_dict

        
@hydra.main(version_base=None, config_path="../config/clean", config_name="base")
def run(conf: DictConfig) -> Union[None, float]:

    # multinode requires this set in submit script
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(conf.experiment.port)

    # Fixes bug in https://github.com/wandb/wandb/issues/1525
    os.environ["WANDB_START_METHOD"] = "thread"


    exp = Experiment(conf=conf)
    if conf.experiment.mode == 'train':
        metrics_dict = exp.start_training()
    else:
        raise ValueError(f'Unknown mode: {conf.experiment.mode}')
    
if __name__ == '__main__':
    run()