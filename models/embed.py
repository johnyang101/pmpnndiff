import torch, torch.nn as nn
import utils.diffusion_utils as diff_u

class TimestepEmbedding(nn.Module): #TODO: Move this to a new file.
    
    def __init__(self, max_len=3000, T=1000, pos_embed_size=64, output_embed_size=20, input_size=20, **kwargs):
        super(TimestepEmbedding, self).__init__()
        self.pos_embed_size = pos_embed_size
        self.output_embed_size = output_embed_size #For finetuning, the output_embed_size must be 20 for now...
        self.max_len = max_len
        
        # positional embedding for position in sequence
        self.register_buffer('pos_embedding_N', torch.tensor(
            diff_u.positional_embedding(max_len, pos_embed_size)))
        
        # Positional embedding for time step.
        self.register_buffer('pos_embedding_T', torch.tensor(
            diff_u.positional_embedding(T + 5, pos_embed_size))) #We add 5 to T for a buffer.

        self.output_embedding = nn.Linear(
            pos_embed_size * 2 + input_size, output_embed_size, bias=False)
        
    def forward(self, B: int, t: torch.Tensor, N: int, seq: torch.Tensor):
        """forward runs the embedding module on a batch of inputs.
        
        Args:
            B, t, N : batchsize, time step, chain size
            seq is Tensor representing sequence to be embedding of size [B, N, input_size]
        Returns:
            embeddings of dimension [B, N, output_embed_size]
        """
        embed_N = torch.tile(self.pos_embedding_N.unsqueeze(0), [B, 1, 1])[:, :N, :] # only need to embed first N positions, 
                                                                              # unsure about backwards compatibility with non GVP, should be fine given before was using fixed N for both max_len and sequence length. 
        embed_T = torch.tile(self.pos_embedding_T[t][:, None, :], [1, N, 1])

        all_embed = torch.concat(
            [embed_N, embed_T, seq], dim=-1).type(torch.float32)

        return self.output_embedding(all_embed) 