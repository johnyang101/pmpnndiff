import torch
import sklearn
import sklearn.preprocessing
import models.pmpnn_utils
from torchtyping import TensorType

'''
Classes store data as instance variables, 
making integration of models with more data structures easily
compatible with diffusion utils.
'''

'''
We register all attributes as buffers so that they automatically move devices with the Obj class device is moved.
'''

class Obj(torch.nn.Module):
    def __init__(self, sequence: TensorType['B', 'N']):
        super().__init__()
        self.register_buffer('seq', sequence)
        x_start = sequence.unsqueeze(0) if len(sequence.shape) == 1 else sequence # We expect batch dimension.
        self.register_buffer('x_start', x_start)
        self.register_buffer('log_x_start', None)
        self.register_buffer('log_x_t', None)
        self.register_buffer('x_t', None)
        if len(sequence.shape) > 2: raise ValueError('Sequence must be 2D.')
        self.B, self.N = sequence.shape if len(sequence.shape) == 2 else (1, sequence.shape[0])
        
    def calculate_accuracy(self):
        """
        Calculates the accuracy by comparing elements in self.x_t with self.x_start 
        at positions where self.mask is True.
        """
        # Ensure the mask is a boolean tensor for indexing
        mask_bool = self.mask.bool()

        # Count the number of elements that are equal in x_t and x_start, considering the mask
        correct = torch.sum(torch.eq(self.x_t[mask_bool], self.x_start[mask_bool]))

        # Calculate the total number of elements considered (as per the mask)
        total = self.mask.sum()

        # Calculate the accuracy
        accuracy = (correct / total).item()

        return accuracy


class PMPNNBatch(Obj): 

    def __init__(self, batch, device):
        self.batch = batch
        X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = models.pmpnn_utils.featurize(batch, device)
        super().__init__(S)
        self.register_buffer('X', X)
        self.register_buffer('S', S)
        self.register_buffer('mask', mask)
        self.lengths = lengths
        self.register_buffer('chain_M', chain_M)
        self.register_buffer('residue_idx', residue_idx)
        self.register_buffer('mask_self', mask_self)
        self.register_buffer('chain_encoding_all', chain_encoding_all)
        self.register_buffer('mask_padded_false', mask) # [1, N])
        self.names = [b['name'] for b in batch]
        self.alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        
    @staticmethod
    def indices_to_string(seq_ind, alphabet, mask, length, pad_token='X'):
        # Convert a sequence of indices to a string using the given alphabet and mask
        seq_chars = [alphabet[idx] if mask[i] else pad_token for i, idx in enumerate(seq_ind[:length])]
        return ''.join(seq_chars)

    def sampled_seq_string(self):
        # Ensure the class attributes are correctly set up
        B, N = self.S.shape
        assert self.S.shape == self.mask.shape, 'seq_ind and mask must have the same shape'
        assert len(self.names) == B, 'names must have the same length as batch_size'

        lengths = self.mask.sum(dim=1).cpu().numpy()  # Calculate lengths from mask
        sampled_seq_string = [self.indices_to_string(self.S[i], self.alphabet, self.mask[i], int(lengths[i])) for i in range(B)]

        return sampled_seq_string
        
def seq_to_one_hot_arr(sequence):
    '''
    @param: sequence - string of single-letter AAs

    output: one-hot encoding of sequence where ROW VECTORS correspond to residue one-
    hot vectors.
    '''

    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
        'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    aastr = ''.join(d.values())
    #aa_letter_to_index = {(aastr[i], i) for i in range(len(aastr))}


    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit([char for char in aastr])

    b = label_binarizer.transform([char for char in sequence])
    assert b.shape == (len(sequence), 20), 'output shape not (len(sequence), 20)'
    return torch.tensor(b).float()

def cath_to_index_arr(cath_dict):
    seq = cath_dict['seq']
    seq = seq_to_one_hot_arr(seq).argmax(dim=1)
    return seq

