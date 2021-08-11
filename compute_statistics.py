import argparse
import logging
import os
import h5py
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle

def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(path, file)

def read_melspec(filepath):
    with h5py.File(filepath, "r") as f:
        melspec = f["melspec"][()]  # n_mels x n_frame
    #import pdb;pdb.set_trace() # Breakpoint
    return melspec

def compute_statistics(src, stat_filepath):
    melspec_scaler = StandardScaler()
        
    filepath_list = list(walk_files(src, '.h5'))
    for filepath in tqdm(filepath_list):
        melspec = read_melspec(filepath)
        #import pdb;pdb.set_trace() # Breakpoint
        melspec_scaler.partial_fit(melspec.T)

    with open(stat_filepath, mode='wb') as f:
        pickle.dump(melspec_scaler, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='./dump/arctic/feat/train')
    parser.add_argument('--stat', type=str, default='./dump/arctic/stat.pkl')
    args = parser.parse_args()

    fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    datafmt = '%m/%d/%Y %I:%M:%S'
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datafmt)

    src = args.src
    stat_filepath = args.stat
    if not os.path.exists(os.path.dirname(stat_filepath)):
        os.makedirs(os.path.dirname(stat_filepath))
    
    compute_statistics(src, stat_filepath)

if __name__ == '__main__':
    main()