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

# Define the parser
parser = argparse.ArgumentParser(description='CSV processing script.')
parser.add_argument('--csv_path', help='Path to CSV file.', type=str)
parser.add_argument('--output_dir', help='Directory to save outputs.', type=str)
parser.add_argument('--device', type=int, default=None, help='gpu id')

def main(args):
    esm_dir = '/data/rsg/chemistry/jyim/.cache/torch'
    torch.hub.set_dir(esm_dir)
    folding_model = esm.pretrained.esmfold_v1().eval()

    if args.device is None:
        available_gpus = ''.join([str(x) for x in GPUtil.getAvailable(order='memory', limit = 8)])
        device = f'cuda:{available_gpus[0]}'
    else:
        device = f'cuda:{args.device}'
    print(f'Using GPU: {device}')
    folding_model = folding_model.to(device)

    # Extract basename from csv_path
    base_name = os.path.basename(args.csv_path).split('.')[0]
    output_subdir = os.path.join(args.output_dir, base_name)
    # Create subdirectory if it doesn't exist
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    # Read CSV data
    with open(args.csv_path, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        csv_data = [row for row in reader]

    random.shuffle(csv_data)

    for i, row in tqdm(enumerate(csv_data)):
        row_id = row[0]  # Unique row ID
        header = row[1]
        string = row[2]
        output_path = os.path.join(output_subdir, f"{header}_{row_id}.pdb")
        if os.path.exists(output_path):
            continue
        print(f'Running {header}_{row_id}')
        with torch.no_grad():
            output = folding_model.infer_pdb(string)
        with open(output_path, "w") as f:
            f.write(output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)