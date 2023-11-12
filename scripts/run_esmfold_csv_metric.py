"""Runs ESMFold over sequences in fasta file.

python scripts/run_esmfold.py --fasta_path /data/rsg/chemistry/jyim/projects/protein_diffusion/notebooks/unconditional_sequences.fa --output_dir /data/scratch/jyim/esmfold_outputs
"""

import argparse
import torch
import esm
import random
import os
import GPUtil
import csv
from tqdm import tqdm
from Bio import PDB
from Bio.PDB.Chain import Chain
from prot_diff.data.utils import *
import utils.analysis_utils as au
import typing as T
import uuid
import pandas as pd

# Define the parser
parser = argparse.ArgumentParser(description='CSV processing script.')
parser.add_argument('--csv_path', help='Path to CSV file.', type=str)
parser.add_argument('--output_dir', help='Directory to save outputs.', type=str)
parser.add_argument('--device', type=int, default=None, help='gpu id')
parser.add_argument('--name', type=str, default='abs_re', help='name of output csv')
parser.add_argument('--test_pdbs_dir', type=str, help='path to test pdbs')
parser.add_argument('--esm_dir', type=str, help='path to esm dir')

def parse_pdb_feats(
        pdb_name: str,
        pdb_path: str,
        scale_factor=1.,
        # TODO: Make the default behaviour read all chains.
        chain_id='A',
    ):
    """
    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        mean_center: whether to mean center atom positions.
    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {
        chain.id: chain
        for chain in structure.get_chains()}
    # print(struct_chains)

    def _process_chain_id(x):
        chain_prot = process_chain(struct_chains[x], x)
        chain_dict = dataclasses.asdict(chain_prot)

        # Process features
        feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
        return parse_chain_feats(
            feat_dict, scale_factor=scale_factor)

    if isinstance(chain_id, str):
        return _process_chain_id(chain_id)
    elif isinstance(chain_id, list):
        return {
            x: _process_chain_id(x) for x in chain_id
        }
    elif chain_id is None:
        return {
            x: _process_chain_id(x) for x in struct_chains
        }
    else:
        raise ValueError(f'Unrecognized chain list {chain_id}')
    
def get_reference_feats(reference_pdb_path: str, chain_id: str) -> T.Dict:
    return parse_pdb_feats('sample', reference_pdb_path, chain_id=chain_id)

def generate_unique_output_path(base_dir, prefix='', extension='.csv'):
    """
    Generates a unique output path for a dataframe.

    Parameters:
    - base_dir (str): The directory where the file will be saved.
    - prefix (str): A prefix for the filename.
    - extension (str): The file extension (including the dot).

    Returns:
    - str: A unique output path for the dataframe.
    """
    while True:
        # Generate a unique identifier
        unique_id = str(uuid.uuid4())
        # Construct the filename and path
        filename = f"{prefix}{unique_id}{extension}"
        output_path = os.path.join(base_dir, filename)
        # Check if a file with this name already exists
        if not os.path.exists(output_path):
            return output_path

def main(args):
    torch.hub.set_dir(args.esm_dir)
    folding_model = esm.pretrained.esmfold_v1().eval()

    # GPU selection logic
    if args.device is None:
        available_gpus = ''.join([str(x) for x in GPUtil.getAvailable(order='memory', limit=8)])
        device = f'cuda:{available_gpus[0]}'
    else:
        device = f'cuda:{args.device}'
    print(f'Using GPU: {device}')
    folding_model = folding_model.to(device)

    # Directory setup
    base_name = os.path.basename(args.csv_path).split('.')[0]
    output_subdir = os.path.join(args.output_dir, base_name)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    # Read and shuffle CSV data
    with open(args.csv_path, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None)  # Skip the headers
        csv_data = [row for row in reader]
    random.shuffle(csv_data)

    # DataFrame columns setup
    data = []  # List to collect row data
    df_output_path = generate_unique_output_path(output_subdir, prefix=f'{args.name}', extension='.csv')

    # Processing loop
    for i, row in tqdm(enumerate(csv_data)):
        row_id, header, string = row[0], row[1], row[2]
        output_path = os.path.join(output_subdir, f"{header}_{row_id}.pdb")

        if not os.path.exists(output_path):
            print(f'Running {header}_{row_id}')
            with torch.no_grad():
                output = folding_model.infer_pdb(string)
            with open(output_path, "w") as f:
                f.write(output)

            feats = get_reference_feats(output_path, 'A') 
            pdb, chid = header.split(".")
            ref_feats = get_reference_feats(os.path.join(args.test_pdbs_dir, pdb + '.pdb'), chain_id=chid)

            d_dict = au.designability_metric(ref_feats, feats, header)
            data.append({'pdb': header, 'id': row_id, 'tm_score': d_dict['tm_score'][0], 'rmsd': d_dict['rmsd'][0]})

    # Create DataFrame from the collected data
    df = pd.DataFrame(data, columns=['pdb', 'id', 'tm_score', 'rmsd'])
    df.to_csv(df_output_path, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)