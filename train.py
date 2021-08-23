# Copyright 2021 Hirokazu Kameoka

import numpy as np
import os
import argparse
import json
import itertools
import logging

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import MultiDomain_Dataset, collate_fn
import convs2s_net as net
#from train import Train

def makedirs_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def prod(N):
    iterable = list(range(0,N))
    return list(itertools.product(iterable,repeat=2)) # (0,0), (0,1), (0,2), (0,3), (1,0), (1,1),...

def perm(N):
    iterable = list(range(0,N))
    return list(itertools.permutations(iterable,2)) # (0,1), (0,2), (0,3), (1,0), (1,2),...

def comb(N):
    iterable = list(range(0,N))
    return list(itertools.combinations(iterable,2)) # (0,1), (0,2), (0,3), (1,2),...

def Train(models, epochs, train_loader, optimizers, model_config, device, model_dir, log_path, snapshot=10, resume=0):
    fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    datafmt = '%m/%d/%Y %I:%M:%S'
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format=fmt, datefmt=datafmt)

    writer = SummaryWriter(os.path.dirname(log_path))

    pw = model_config['pos_weight']
    gw = model_config['gauss_width_da']
    rf = model_config['reduction_factor']
    w_da_init = model_config['w_da']
    multistep = model_config['multistep']
    iml = model_config['identity_mapping']

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for tag in ['enc_src', 'enc_trg', 'dec']:
        checkpointpath = os.path.join(model_dir, '{}.{}.pt'.format(resume,tag))
        if os.path.exists(checkpointpath):
            checkpoint = torch.load(checkpointpath, map_location=device)
            models[tag].load_state_dict(checkpoint['model_state_dict'])
            optimizers[tag].load_state_dict(checkpoint['optimizer_state_dict'])
            print('{} loaded successfully.'.format(checkpointpath))

    n_iter = 0
    print("===================================Start Training===================================")
    logging.info(model_dir)
    for epoch in range(resume+1, epochs+1):
        b = 0
        w_da = w_da_init * np.exp(-epoch/50.0)
        for X_list, mask_list in train_loader:
            n_spk = len(X_list)
            xin = []
            mask = []
            for s in range(n_spk):
                xin.append(torch.tensor(X_list[s]).to(device, dtype=torch.float))
                mask.append(torch.tensor(mask_list[s]).to(device, dtype=torch.float))

            if iml:
                mainloss_mean = 0
                daloss_mean = 0
                for s in range(n_spk):
                    MainLoss, DALoss, A = models['convs2s'].calc_loss(xin[s], xin[s], mask[s], mask[s], s, s, pw, gw, rf)
                    Loss = MainLoss + w_da * DALoss

                    mainloss_mean = mainloss_mean + MainLoss.item()
                    daloss_mean = daloss_mean + DALoss.item()

                    if multistep:
                        for tag in ['enc_src', 'enc_trg']:
                            models[tag].zero_grad()
                            Loss.backward(retain_graph=True)
                            optimizers[tag].step()
                        for tag in ['dec']:
                            models[tag].zero_grad()
                            Loss.backward()
                            optimizers[tag].step()

                    else:
                        for tag in ['enc_src', 'enc_trg', 'dec']:
                            models[tag].zero_grad()
                        Loss.backward()
                        for tag in ['enc_src', 'enc_trg', 'dec']:
                            optimizers[tag].step()

                mainloss_mean = mainloss_mean/n_spk
                daloss_mean = daloss_mean/n_spk
                logging.info('epoch {}, mini-batch {}: IMLoss={:.4f}, DALoss={:.4f}'.format(epoch, b+1, mainloss_mean, w_da*daloss_mean))
                writer.add_scalar('Loss/MainLoss_IM', mainloss_mean, n_iter)
                writer.add_scalar('Loss/DALoss_IM', w_da*daloss_mean, n_iter)

            # List of speaker pairs
            spk_pair_list = perm(n_spk)
            n_spk_pair = len(spk_pair_list)

            mainloss_mean = 0
            daloss_mean = 0
            # Iterate through all speaker pairs
            for m in range(n_spk_pair):
                s0 = spk_pair_list[m][0]
                s1 = spk_pair_list[m][1]
                MainLoss, DALoss, A = models['convs2s'].calc_loss(xin[s0], xin[s1], mask[s0], mask[s1], s0, s1, pw, gw, rf)
                Loss = MainLoss + w_da * DALoss

                mainloss_mean = mainloss_mean + MainLoss.item()
                daloss_mean = daloss_mean + DALoss.item()

                if multistep:
                    for tag in ['enc_src', 'enc_trg']:
                        models[tag].zero_grad()
                        Loss.backward(retain_graph=True)
                        optimizers[tag].step()
                    for tag in ['dec']:
                        models[tag].zero_grad()
                        Loss.backward()
                        optimizers[tag].step()

                else:
                    for tag in ['enc_src', 'enc_trg', 'dec']:
                        models[tag].zero_grad()
                    Loss.backward()
                    for tag in ['enc_src', 'enc_trg', 'dec']:
                        optimizers[tag].step()

            mainloss_mean = mainloss_mean/n_spk_pair
            daloss_mean = daloss_mean/n_spk_pair

            logging.info('epoch {}, mini-batch {}: MainLoss={:.4f}, DALoss={:.4f}'.format(epoch, b+1, mainloss_mean, w_da*daloss_mean))
            writer.add_scalar('Loss/MainLoss', mainloss_mean, n_iter)
            writer.add_scalar('Loss/DALoss', w_da*daloss_mean, n_iter)

            n_iter += 1
            b += 1

        if epoch % snapshot == 0:
            for tag in ['enc_src', 'enc_trg', 'dec']:
                #print('save {} at {} epoch'.format(tag, epoch))
                torch.save({'epoch': epoch,
                            'model_state_dict': models[tag].state_dict(),
                            'optimizer_state_dict': optimizers[tag].state_dict()},
                            os.path.join(model_dir, '{}.{}.pt'.format(epoch, tag)))

    print("===================================Training Finished===================================")

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
    parser.add_argument('--num_layers', '-nl', type=int, default=8, help='Number of layers in each network')
    parser.add_argument('--lrate', '-lr', default='5e-05', type=float, help='learning rate')
    parser.add_argument('--w_da', '-wd', default='2000.0', type=float, help='regularization weight for DAL')
    parser.add_argument('--pos_weight', '-pw', default='1.0', type=float, help='Weight for positional encoding')
    parser.add_argument('--dropout_ratio', '-dr', default='0.1', type=float, help='dropout ratio')
    parser.add_argument('--gauss_width_da', '-gda', default='0.3', type=float, help='Width of Gaussian for DAL')
    parser.add_argument('--identity_mapping', '-iml', default=1, type=int, help='{0: not include 1: include} IML')
    parser.add_argument('--reduction_factor', '-rf', default=4, type=int, help='Reduction factor')
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
    lrate = args.lrate
    w_da = args.w_da
    pos_weight = args.pos_weight
    dropout_ratio = args.dropout_ratio
    gauss_width_da = args.gauss_width_da
    identity_mapping = bool(args.identity_mapping)
    reduction_factor = args.reduction_factor
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
        'lrate': lrate,
        'w_da': w_da,
        'pos_weight': pos_weight,
        'dropout_ratio': dropout_ratio,
        'gauss_width_da': gauss_width_da,
        'identity_mapping': identity_mapping, 
        'reduction_factor': reduction_factor,
        'multistep': multistep, 
        'epochs': epochs,
        'BatchSize': batch_size,
        'n_spk': n_spk,
        'spk_list': spk_list
    }

    model_dir = os.path.join(args.model_rootdir, args.experiment_name)
    makedirs_if_not_exists(model_dir)
    log_path = os.path.join(args.log_dir, args.experiment_name, 'train_{}.log'.format(args.experiment_name))
    
    # Save configuration as a json file
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as outfile:
        json.dump(model_config, outfile, indent=4)

    models = {
        'enc_src' : net.SrcEncoder1(num_mels*reduction_factor,n_spk,hdim,zdim,kdim,num_layers,dropout_ratio),
        'enc_trg' : net.TrgEncoder1(num_mels*reduction_factor,n_spk,hdim,zdim,kdim,num_layers,dropout_ratio),
        'dec' : net.Decoder1(zdim*2,n_spk,hdim,num_mels*reduction_factor,mdim,num_layers,dropout_ratio)
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
                              shuffle=True,
                              num_workers=0,
                              #num_workers=os.cpu_count(),
                              drop_last=True,
                              collate_fn=collate_fn)
    Train(models, epochs, train_loader, optimizers, model_config, device, model_dir, log_path, snapshot, resume)


if __name__ == '__main__':
    main()