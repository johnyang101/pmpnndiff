import math
import torch
import torch.nn.functional as F
import torch.distributions as dists
from torchtyping import TensorType

from models.pmpnn_lms import Generic_LM
from data.data_objs import PMPNNBatch
from models.pmpnn import PMPNN_Baseline_Diff

import models.diffusion_utils as du

class Generic_Diff_LM(Generic_LM):
    def __init__(self, cfg, debug=False):
        super().__init__(cfg)
        self.debug = debug
        self.num_classes = cfg.num_classes
        self.non_abs_classes = self.num_classes - 1 if self.cfg.model.absorbing else self.num_classes
        self._denoise_fn = self._init_denoise_fn(cfg.model)
        if self.cfg.model.ft:
            self._denoise_fn = self.load_ft_dict(self._denoise_fn)
        
    def _init_denoise_fn(self, model_conf): #TODO: Write this.
        return PMPNN_Baseline_Diff(**model_conf)
        
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
    
class UNL_Absorbing_Diff_LM(Generic_Diff_LM):
    def __init__(self, cfg, debug=False):
        super().__init__(cfg)
        
        self.num_timesteps = cfg.num_timesteps if not self.debug else 10
        
        self.num_classes = cfg.num_classes #Expects alphabet size + 1 for mask
        self.mask_id = self.num_classes - 1
        self.loss_type = cfg.loss_type
        self.mask_schedule = cfg.mask_schedule
        assert self.mask_schedule in ['random', 'fixed']

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))        

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
        elif self.mask_schedule == 'fixed':
            x_t, x_0_ignore, mask = self.q_sample_mlm(x_0=x_0, t=t)
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
    
    def _prepare_sample(self, obj):
        b = obj.B
        device = self.device
        obj.x_t = torch.ones_like(obj.x_start, device=self.device).long() * self.mask_id
        unmasked = ~obj.mask.bool()
        return b, device, unmasked

    def _update_timestep(self, b, device, t):
        print(f'Sample timestep {t:4d}', end='\r')
        return torch.full((b,), t, device=device, dtype=torch.long)

    def _update_mask_and_sample(self, changes, unmasked, x_0_hat, obj):
        # update mask with changes
        unmasked = torch.bitwise_or(unmasked, changes)
        obj.x_t[changes] = x_0_hat[changes]
        return unmasked

    def sample(self, obj):
        b, device, unmasked = self._prepare_sample(obj)
        sample_steps = list(range(1, self.cfg.num_timesteps))

        for t in reversed(sample_steps):
            t = self._update_timestep(b, device, t)
            changes = torch.rand(obj.x_t.shape, device=device) < 1 / t.float().unsqueeze(-1)
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            x_0_logits = self._denoise_fn(obj, t=t) / self.cfg.temp
            x_0_hat = dists.Categorical(logits=x_0_logits).sample().long()
            unmasked = self._update_mask_and_sample(changes, unmasked, x_0_hat, obj)

        assert (torch.where(obj.x_t == self.mask_id, torch.ones_like(obj.x_t), torch.zeros_like(obj.x_t)) * obj.mask).sum() == 0, 'Mask ID still in x_t'
        accuracy = obj.calculate_accuracy()
        return obj, accuracy
    
    def purity_sample(self, obj, stride=1, max_sample_ratio_per_step=0.05, prior_weight=0.5):
        b, device, unmasked = self._prepare_sample(obj)
        sample_steps = list(range(1, self.num_timesteps, stride))
        X, S, mask, chain_M, residue_idx, chain_encoding_all = obj.X, obj.S, obj.mask, obj.chain_M, obj.residue_idx, obj.chain_encoding_all
        
        h_V, h_E, E_idx = self._denoise_fn.encode(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        max_sample_per_step = round((obj.N * max_sample_ratio_per_step))
        sampled = torch.zeros(obj.B)
        to_sample = obj.N
        
        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t_tensor = self._update_timestep(b, device, t)

            one_hot_x_t = F.one_hot(obj.x_t, self.num_classes)
            h_S = self._denoise_fn.W_s(obj.B, t_tensor, obj.N, one_hot_x_t)
            log_x_recon = self._denoise_fn.decode(mask, chain_M, h_V, h_E, E_idx, h_S).permute(0, 2, 1)

            score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
            normalized_score = score / (score.max(dim=1, keepdim=True).values + 1e-10)
            adjusted_prob = ((1 + normalized_score * prior_weight).unsqueeze(1) * log_x_recon).softmax(dim=1)
            log_prob = adjusted_prob.log().clamp(-70, 0)

            x_0_dist = dists.Categorical(logits=log_prob)
            sampled_seq = x_0_dist.sample().long()

            updated_x_t = obj.x_t.clone()
            normalized_score[obj.x_t != self.mask_id] = 0

            for i in range(obj.B):
                available_to_sample = to_sample - sampled[i].item()
                n_sample = min(round(available_to_sample), max_sample_per_step)
                if available_to_sample - n_sample == 1:
                    n_sample = round(available_to_sample)

                if n_sample > 0:
                    selected_indices = torch.multinomial(normalized_score[i], n_sample)
                    updated_x_t[i][selected_indices] = sampled_seq[i][selected_indices]
                    num_sampled = (updated_x_t[i] != self.mask_id).sum().item() - (obj.x_t[i] != self.mask_id).sum().item()
                    sampled[i] += num_sampled

            obj.x_t = updated_x_t

        assert (torch.where(obj.x_t == self.mask_id, torch.ones_like(obj.x_t), torch.zeros_like(obj.x_t)) * obj.mask).sum() == 0, 'Mask ID still in x_t'
        accuracy = obj.calculate_accuracy()
        return obj, accuracy


    def validation_step(self, batch, batch_idx):
        obj = self.get_obj(batch)
        obj.to(self.device)
        
        obj = self.sample(obj)

        acc = torch.sum(torch.eq(obj.x_t[obj.mask.bool()], obj.x_start[obj.mask.bool()])) / obj.mask.sum()
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=obj.B)
        return acc
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)