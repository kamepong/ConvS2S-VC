import numpy as np
import os
import argparse
import torch
from torch import optim
import convs2s_net as net
import json

from torch.utils.data import DataLoader
from dataset import MultiDomain_Dataset, collate_fn
from train import Train

def makedirs_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def main():
    parser = argparse.ArgumentParser(description='ConvS2S-VC')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('-ddir', '--data_rootdir', type=str, default='./dump/arctic/norm_feat/train',
                        help='root data folder that contains the normalized features')
    parser.add_argument('--epochs', '-epoch', default=100, type=int, help='number of epochs to learn')
    parser.add_argument('--snapshot', '-snap', default=10, type=int, help='snapshot interval')
    parser.add_argument('--batch_size', '-batch', type=int, default=16, help='Batch size')
    parser.add_argument('--num_mels', '-nm', type=int, default=80, help='number of mel channels')
    parser.add_argument('--zdim', '-zd', type=int, default=512, help='latent space dimension')
    parser.add_argument('--kdim', '-kd', type=int, default=512, help='middle layer dimension of encoders')
    parser.add_argument('--mdim', '-md', type=int, default=512, help='middle layer dimension of decoder')
    parser.add_argument('--hdim', '-hd', type=int, default=32, help='speaker embedding dimension')
    parser.add_argument('--num_layers', '-nl', type=int, default=4, help='Number of layers in each dilated CNN block')
    parser.add_argument('--num_blocks', '-nb', type=int, default=2, help='Number of blocks')
    parser.add_argument('--lrate', '-lr', default='5e-05', type=float, help='learning rate')
    parser.add_argument('--w_da', '-wd', default='2000.0', type=float, help='regularization weight for DAL')
    parser.add_argument('--pos_weight', '-pw', default='1.0', type=float, help='Weight for positional encoding')
    parser.add_argument('--dropout_ratio', '-dr', default='0.1', type=float, help='dropout ratio')
    parser.add_argument('--gauss_width_da', '-gda', default='0.3', type=float, help='Width of Gaussian for DAL')
    parser.add_argument('--identity_mapping', '-iml', default=1, type=int, help='{0: not include 1: include} IML')
    parser.add_argument('--reduction_factor', '-rf', default=4, type=int, help='Reduction factor')
    parser.add_argument('--normtype', '-norm', default='WN', type=str, help='normalization type: CBN, WN')
    parser.add_argument('--multistep', '-ms', default=1, type=int, help='Multistep parameter update')
    parser.add_argument('--resume', '-res', type=int, default=0, help='Checkpoint to resume training')
    parser.add_argument('--model_rootdir', '-mdir', type=str, default='./model/arctic/', help='model file directory')
    parser.add_argument('--log_dir', '-ldir', type=str, default='./logs/arctic/', help='log file directory')
    parser.add_argument('--experiment_name', '-exp', default='experiment1', type=str, help='experiment name')
    args = parser.parse_args()

    # Set up GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Configuration for ConvS2S
    num_mels = args.num_mels
    zdim = args.zdim
    kdim = args.kdim
    mdim = args.mdim
    hdim = args.hdim
    num_layers = args.num_layers
    num_blocks = args.num_blocks
    lrate = args.lrate
    w_da = args.w_da
    pos_weight = args.pos_weight
    dropout_ratio = args.dropout_ratio
    gauss_width_da = args.gauss_width_da
    identity_mapping = bool(args.identity_mapping)
    reduction_factor = args.reduction_factor
    normtype = args.normtype
    multistep = bool(args.multistep)
    epochs = args.epochs
    batch_size = args.batch_size
    snapshot = args.snapshot
    resume = args.resume

    data_rootdir = args.data_rootdir
    spk_list = sorted(os.listdir(data_rootdir))
    n_spk = len(spk_list)
    melspec_dirs = [os.path.join(data_rootdir,spk) for spk in spk_list]

    model_config = {
        'num_mels': num_mels,
        'zdim': zdim,
        'kdim': kdim,
        'mdim': mdim,
        'hdim': hdim,
        'num_layers': num_layers,
        'num_blocks': num_blocks,
        'lrate': lrate,
        'w_da': w_da,
        'pos_weight': pos_weight,
        'dropout_ratio': dropout_ratio,
        'gauss_width_da': gauss_width_da,
        'identity_mapping': identity_mapping, 
        'reduction_factor': reduction_factor,
        'normtype': normtype,
        'multistep': multistep, 
        'epochs': epochs,
        'BatchSize': batch_size,
        'n_spk': n_spk,
        'spk_list': spk_list
    }

    model_dir = os.path.join(args.model_rootdir, args.experiment_name)
    makedirs_if_not_exists(model_dir)
    log_path = os.path.join(args.log_dir, 'train_{}.log'.format(args.experiment_name))
    
    # Save configuration as a json file
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as outfile:
        json.dump(model_config, outfile, indent=4)

    #import pdb; pdb.set_trace()
    models = {
        'enc_src' : net.SrcEncoder1(num_mels*reduction_factor,n_spk,hdim,zdim,kdim,num_layers,num_blocks,normtype),
        'enc_trg' : net.TrgEncoder1(num_mels*reduction_factor,n_spk,hdim,zdim,kdim,num_layers,num_blocks,normtype),
        'dec' : net.Decoder1(zdim*2,n_spk,hdim,num_mels*reduction_factor,mdim,num_layers,num_blocks,normtype)
    }
    models['convs2s'] = net.ConvS2S(models['enc_src'], models['enc_trg'], models['dec'])

    optimizers = {
        'enc_src' : optim.Adam(models['enc_src'].parameters(), lr=lrate, betas=(0.9,0.999)),
        'enc_trg' : optim.Adam(models['enc_trg'].parameters(), lr=lrate, betas=(0.9,0.999)),
        'dec' : optim.Adam(models['dec'].parameters(), lr=lrate, betas=(0.9,0.999))
    }

    for tag in ['enc_src', 'enc_trg', 'dec']:
        models[tag].to(device).train(mode=True)

    train_dataset = MultiDomain_Dataset(*melspec_dirs)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              #num_workers=os.cpu_count(),
                              drop_last=False,
                              collate_fn=collate_fn)
    Train(models, epochs, train_loader, optimizers, model_config, device, model_dir, log_path, snapshot, resume)


if __name__ == '__main__':
    main()