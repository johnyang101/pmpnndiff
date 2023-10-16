import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Dict, List, Optional, Tuple, Union
from torchtyping import TensorType, patch_typeguard
from omegaconf import DictConfig

import pmpnn_utils as pu
import models.embed

'''
Naming scheme for models:
Name: [Model Type]_[Model Variant]_[Task]

'''   
    
class PMPNN_Baseline_CPD(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1, disable_random_decoding_order=False, **kwargs):
        super(PMPNN_Baseline_CPD, self).__init__()

        self.disable_random_decoding_order = disable_random_decoding_order
        
        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = pu.ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            pu.EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            pu.DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = pu.gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
        
        return h_V, h_E, E_idx

    def decode(self, mask, chain_M, h_V, h_E, E_idx, h_S):
        device=h_V.device
        
        # Concatenate sequence embeddings for autoregressive decoder
        h_ES = pu.cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = pu.cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = pu.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
        if self.disable_random_decoding_order:
            decoding_order: TensorType['B', 'N'] = torch.tile(torch.arange(0, chain_M.shape[1], device=device), (chain_M.shape[0], 1))
        else:
            decoding_order: TensorType['B', 'N'] = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse: TensorType['B', 'N', 'N'] = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = pu.cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        return logits

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        
        h_V, h_E, E_idx = self.encode(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        h_S = self.W_s(S)
        logits = self.decode(mask, chain_M, h_V, h_E, E_idx, h_S)
        # log_probs = F.log_softmax(logits, dim=-1)
        return logits
    
    def sample(self, X, deco_ranking, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, bias_by_res=None,):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = pu.gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(deco_ranking))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        #chain_mask_combined = chain_mask*chain_M_pos 
        omit_AA_mask_flag = omit_AA_mask != None


        h_EX_encoder = pu.cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = pu.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:,t_] #[B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
            mask_gathered = torch.gather(mask, 1, t[:,None]) #[B]
            bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,21))[:,0,:] #[B, 21]
            if (mask_gathered==0).all(): #for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:,None]) #Samples mask tokens first.
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = pu.cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:,None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = pu.cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
                    pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
                    probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:,None,None].repeat(1,1,21), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:,None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
            S.scatter_(1, t[:,None], S_t)
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict

class PMPNN_Baseline_Diff(PMPNN_Baseline_CPD):
    
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1,
        num_classes: int = 20,
        embedding_cfg: Union[Dict, DictConfig] = {}, absorbing=False,
        **kwargs):
        
        assert embedding_cfg, 'Must pass in embedding conf'
        # edge_features = edge_features + embedding_cfg['output_embed_size']
        
        super().__init__(
            num_letters, 
            node_features, 
            edge_features,
            hidden_dim,
            num_encoder_layers,
            num_decoder_layers,
            vocab,
            k_neighbors,
            augment_eps,
            dropout
        )
        self.W_s = models.embed.TimestepEmbedding(**embedding_cfg) 
        self.num_classes = num_classes 
        self.max_len = embedding_cfg['max_len']
        
        if absorbing:
            self.W_out = nn.Linear(hidden_dim, self.num_classes - 1, bias=True)

        assert self.num_classes == num_letters, 'PMPNN Denoise Fn must have same number of classes as num_letters'
        
    def decode(self, mask, chain_M, h_V, h_E, E_idx, h_S):
        """
        Decodes without causal attention mask.
        """
        
        h_ES = pu.cat_neighbors_nodes(h_S, h_E, E_idx)
        chain_M = chain_M*mask #update chain_M to include missing regions
        for layer in self.decoder_layers:
            h_ESV = pu.cat_neighbors_nodes(h_V, h_ES, E_idx)
            # h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        return logits
    
    def forward(self, obj, t):
        
        X, S, mask, chain_M, residue_idx, chain_encoding_all = obj.X, obj.S, obj.mask, obj.chain_M, obj.residue_idx, obj.chain_encoding_all
        
        h_V, h_E, E_idx = self.encode(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        
        x_t = obj.x_t
        one_hot_x_t: TensorType[obj.B, obj.N, self.num_classes] = F.one_hot(x_t, self.num_classes)
        
        #Embed timestep in sequence.
        h_S = self.W_s(obj.B, t, obj.N, one_hot_x_t)

        logits = self.decode(mask, chain_M, h_V, h_E, E_idx, h_S)
        return logits