import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.utils import StructureDatasetJSONL, StructureLoader
from data.utils import StructureDatasetPDB, PDB_dataset, loader_pdb, build_training_clusters, worker_init_fn, get_pdbs

class Generic_PMPNN_DM(pl.LightningDataModule):
    def __init__(self, cfg, debug, truncate=None):
        super().__init__()
        self.cfg = cfg
        self.debug = debug
        self.truncate = 100 if debug and truncate is None else truncate
        self.memorize = True if self.truncate == 1 else False
        self.batch_size = cfg.batch_size
        print(f'Debug: {self.debug}')
        print(f"Batch size: {self.batch_size}")
        
    def train_dataloader(self) -> StructureLoader:
        return StructureLoader(self.dataset_train, batch_size=self.batch_size) 
    
    def val_dataloader(self) -> StructureLoader:
        return StructureLoader(self.dataset_val, batch_size=self.batch_size) 
    
    def test_dataloader(self, shuffle=True) -> StructureLoader:
        return StructureLoader(self.dataset_test, batch_size=self.batch_size, shuffle=shuffle)
    
    def predict_dataloader(self) -> StructureLoader:
        return StructureLoader(self.dataset_test, batch_size=self.batch_size)

class PMPNN_JSONL_DM(Generic_PMPNN_DM):
    
    def __init__(self, cfg, debug, truncate=None):
        super().__init__(cfg, debug, truncate)
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None: #TODO: Standardize self.cfg.train_jsonl_path, self.cfg.valid_jsonl_path, self.cfg.test_jsonl_path
            self.dataset_train = StructureDatasetJSONL(self.cfg.train_jsonl_path, truncate=self.truncate, max_length=self.cfg.max_protein_length, alphabet=self.cfg.alphabet)
            self.dataset_val = self.dataset_train if self.memorize else StructureDatasetJSONL(self.cfg.valid_jsonl_path, truncate=self.truncate, max_length=self.cfg.max_protein_length, alphabet=self.cfg.alphabet)
        if stage == 'test' or stage == 'predict' or stage is None:
            self.dataset_test = StructureDatasetJSONL(self.cfg.test_jsonl_path, truncate=self.truncate, max_length=self.cfg.max_protein_length, alphabet=self.cfg.alphabet)    

class PMPNN_MC_DM(Generic_PMPNN_DM):
    def __init__(self, cfg, debug, truncate=None):
        super().__init__(cfg, debug, truncate)
    
    def _process_pdb_dataset(self, cluster):
        pdb_dataset = PDB_dataset(list(cluster.keys()), loader_pdb, cluster, self.cfg.params)
        pdb_loader = DataLoader(pdb_dataset, worker_init_fn=worker_init_fn, batch_size=1, shuffle=True, pin_memory=False, num_workers=80)
        pdb_dict = get_pdbs(pdb_loader)
        return StructureDatasetPDB(pdb_dict, truncate=self.truncate, max_length=self.cfg.max_protein_length, alphabet=self.cfg.alphabet)
    
    def _build_clusters(self):
        train, valid, test = build_training_clusters(self.cfg.params, self.debug)
        return train, valid, test
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = torch.load(self.cfg.train_path, map_location='cpu') if not self.debug else torch.load(self.cfg.val_path)
                                                                    #We do this because dataset_train is very large.
            self.dataset_val = torch.load(self.cfg.val_path, map_location='cpu')
        if stage == 'test' or stage == 'predict' or stage is None:        
            self.dataset_test = torch.load(self.cfg.test_path, map_location='cpu')
