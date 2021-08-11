import argparse
import joblib
import logging
import os

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle

def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(path, file)

def melspec_transform(melspec, scaler):
    # melspec.shape: (n_freq, n_time)
    # scaler.transform assumes the first axis to be the time axis
    melspec = scaler.transform(melspec.T)
    #import pdb;pdb.set_trace() # Breakpoint
    melspec = melspec.T
    return melspec

def normalize_features(src_filepath, dst_filepath, melspec_transform):
    try:
        with h5py.File(src_filepath, "r") as f:
            melspec = f["melspec"][()]
        melspec = melspec_transform(melspec)
        
        if not os.path.exists(os.path.dirname(dst_filepath)):
            os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
        with h5py.File(dst_filepath, "w") as f:
            f.create_dataset("melspec", data=melspec)
        
        #logging.info(f"{dst_filepath}...[{melspec.shape}].")
        return melspec.shape

    except:
        logging.info(f"{dst_filepath}...failed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str,
                        default='./dump/arctic/feat/train',
                        help='data folder that contains the raw features extracted from VoxCeleb2 Dataset')
    parser.add_argument('--dst', type=str, default='./dump/arctic/norm_feat/train',
                        help='data folder where the normalized features are stored')
    parser.add_argument('--stat', type=str, default='./dump/arctic/stat.pkl', 
                        help='state file used for normalization')
    parser.add_argument('--ext', type=str, default='.h5')
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    ext = args.ext
    stat_filepath = args.stat

    fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    datafmt = '%m/%d/%Y %I:%M:%S'
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datafmt)
    
    melspec_scaler = StandardScaler()
    if os.path.exists(stat_filepath):
        with open(stat_filepath, mode='rb') as f:
            melspec_scaler = pickle.load(f)
        print('Loaded mel-spectrogram statistics successfully.')
    else:
        print('Stat file not found.')
        
    root = src
    fargs_list = [
        [
            f,
            f.replace(src, dst),
            lambda x: melspec_transform(x, melspec_scaler),
        ]
        for f in walk_files(root, ext)
    ]
    
    #import pdb;pdb.set_trace() # Breakpoint
    # debug
    #normalize_features(*fargs_list[0])
    # test
    #results = joblib.Parallel(n_jobs=-1)(
    #    joblib.delayed(normalize_features)(*f) for f in tqdm(fargs_list)
    #)
    results = joblib.Parallel(n_jobs=16)(
        joblib.delayed(normalize_features)(*f) for f in tqdm(fargs_list)
    )

if __name__ == '__main__':
    main()