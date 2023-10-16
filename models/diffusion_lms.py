from models.pmpnn_lms import *
from models.diffusion_utils import *
from data.data_objs import *
import torch.distributions as dists

class Generic_Diff_LM(Generic_LM):
    def __init__(self, cfg, debug=False):
        super().__init__(cfg)
        self.debug = debug
        self.model_type = self.cfg.model_type
        self.token_type = self.cfg.token.type
        self.num_classes = cfg.num_classes
        self.non_abs_classes = self.num_classes - 1 if self.cfg.model.absorbing else self.num_classes
        self._denoise_fn = self._init_denoise_fn(cfg.model)
        if self.cfg.model.ft:
            self._denoise_fn = self.load_ft_dict(self._denoise_fn)
        
        self.shape = cfg.shape #TODO: Address shape, we only use for sampling.
        
    def _init_denoise_fn(self, model_conf): #TODO: Write this.
        ...
        
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
        
        # self.latent_shape = cfg.latent_shape
        # self.emb_dim = cfg.emb_dim
        # self.codebook_size = cfg.codebook_size

        # self.latent_emb_dim = cfg.emb_dim
        self.shape = cfg.shape #tuple(cfg.latent_shape)
        self.num_timesteps = cfg.num_timesteps if not self.debug else 10
        
        self.num_classes = cfg.num_classes #Expects alphabet size + 1 for mask
        self.mask_id = self.num_classes - 1
        self.n_samples = cfg.n_samples
        self.loss_type = cfg.loss_type
        self.mask_schedule = cfg.mask_schedule
        self.aux_weight = cfg.aux_weight
        
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))

        assert self.mask_schedule in ['random', 'fixed']
        
        self.temp = cfg.temp

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
        
        #TODO: Check why we have `ignore_index=-1`, might be a small bug.
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

    def sample(self, obj):
        b = obj.B
        device = self.device
        obj.x_t = torch.ones_like(obj.x_start, device=self.device).long() * self.mask_id
        unmasked = ~obj.mask.bool()
        sample_steps = list(range(1, self.cfg.num_timesteps)) #NOTE: We don't add +1 bc of TimestepEmbedding bugs.

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            changes = torch.rand(obj.x_t.shape, device=device) < 1/t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            x_0_logits = self._denoise_fn(obj, t=t)
            # scale by temperature
            x_0_logits = x_0_logits / self.cfg.temp
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            obj.x_t[changes] = x_0_hat[changes]
        assert (torch.where(obj.x_t == self.mask_id, torch.ones_like(obj.x_t), torch.zeros_like(obj.x_t)) * obj.mask).sum() == 0, 'Mask ID still in x_t'
        return obj
    
    def purity_sample(self, obj, stride=1, max_sample_ratio_per_step=0.05, prior_weight=0.5): #TODO: Combine sample and purity_sample
        b = obj.B
        device = self.device
        obj.x_t = torch.ones_like(obj.x_start, device=self.device).long() * self.mask_id
        unmasked = ~obj.mask.bool()
        sample_steps = list(range(1, self.num_timesteps, stride)) #NOTE: We don't add +1 bc of TimestepEmbedding bugs.
        X, S, mask, chain_M, residue_idx, chain_encoding_all = obj.X, obj.S, obj.mask, obj.chain_M, obj.residue_idx, obj.chain_encoding_all
        
        h_V, h_E, E_idx = self._denoise_fn.encode(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        
        max_sample_per_step = round((obj.N * max_sample_ratio_per_step))
        
        sampled = torch.zeros(obj.B)
        to_sample = obj.N
        
        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            
            x_t = log_x_idx = obj.x_t
            one_hot_x_t: TensorType[obj.B, obj.N, self.num_classes] = F.one_hot(x_t, self.num_classes)
            
            # if self.ablate_struct:
            #     h_V, h_E = self.ablate_struct_for_masked_tokens(x_t, h_V, h_E)
            
            #Embed timestep in sequence.
            h_S = self._denoise_fn.W_s(obj.B, t, obj.N, one_hot_x_t)

            x_0_logits = self._denoise_fn.decode(mask, chain_M, h_V, h_E, E_idx, h_S).permute(0, 2, 1)  

            # x_0_logits = self._denoise_fn(obj, t=t)
            log_x_recon = x_0_logits

            score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
            score /= (score.max(dim=1, keepdim=True).values + 1e-10)

            # probability adjust parameter, prior_weight: 'r' in Equation.11 of Improved VQ-Diffusion
            prob = ((1 + score * prior_weight).unsqueeze(1) * log_x_recon).softmax(dim=1)
            prob = prob.log().clamp(-70, 0)

            x_0_logits = prob / self.cfg.temp
            x_0_dist = dists.Categorical(
                logits=x_0_logits.permute(0, 2, 1))
            x_0_hat = out_idx = x_0_dist.sample().long()

            # out = self.log_sample_categorical(prob)
            # out_idx = log_onehot_to_index(out)

            out2_idx = log_x_idx.clone()
            _score = score.clone()
            if _score.sum() < 1e-6:
                _score += 1
            _score[log_x_idx != self.mask_id] = 0

            for i in range(obj.B):
                n_sample = min(round((to_sample - sampled[i].item())), max_sample_per_step)
                if to_sample - sampled[i] - n_sample == 1:
                    n_sample = round(to_sample - sampled[i].item())
                if n_sample <= 0:
                    continue
                # to_sample -= n_sample

                sel = changes = torch.multinomial(_score[i], n_sample)
                            # where to unmask
                # changes = torch.rand(obj.x_t.shape, device=device) < 1/t.float().unsqueeze(-1)
                # don't unmask somewhere already unmasked #NOTE: This doesn't allow model to make edits.
                # changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
                # # update mask with changes
                # unmasked = torch.bitwise_or(unmasked, changes)
                
                out2_idx[i][sel] = out_idx[i][sel]
                sampled[i] += ((out2_idx[i] != self.mask_id).sum() - (log_x_idx[i] != self.mask_id).sum()).item()
                obj.x_t[i] = out2_idx[i]
            # out = index_to_log_onehot(out2_idx, self.num_classes)
        assert (torch.where(obj.x_t == self.mask_id, torch.ones_like(obj.x_t), torch.zeros_like(obj.x_t)) * obj.mask).sum() == 0, 'Mask ID still in x_t'
        acc = torch.sum(torch.eq(obj.x_t[obj.mask.bool()], obj.x_start[obj.mask.bool()])) / obj.mask.sum()
        # self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=obj.B)
        return obj, acc

    def validation_step(self, batch, batch_idx):
        obj = self.get_obj(batch)
        obj.to(self.device)
        
        obj = self.sample(obj)

        acc = torch.sum(torch.eq(obj.x_t[obj.mask.bool()], obj.x_start[obj.mask.bool()])) / obj.mask.sum()
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=obj.B)
        return acc
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def sample_and_process_sequences(self, batch):
        obj = self.get_obj(batch)
        obj.to(self.device)
        if isinstance(obj, PMPNNBatch):
            B, N = obj.S.shape
            
            names = [b['name'] for b in batch]
            assert len(names) == B, 'len(names) != B'

            header_sequence_tuples = []

            for _ in range(self.cfg.pred_cfg.num_pred_samples):
                sampled_obj = self.sample(obj)
                z_0 = sampled_obj.x_t
                
                formatted_sequences = au.format_sequences(z_0, obj.mask, obj.lengths, names)
                assert len(formatted_sequences) == len(names), 'Number of sequences should be equal to number of names.'
                
                header_sequence_tuples.extend([(name, seq) for name, seq in zip(names, formatted_sequences)])
        else:
            raise ValueError(f'Unknown obj type: {type(obj)}')
        
        return header_sequence_tuples