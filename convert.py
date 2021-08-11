import os
import argparse
import torch
import json
import numpy as np
import re
import pickle
from tqdm import tqdm
import yaml

import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler

import convs2s_net as net
from extract_features import logmelfilterbank

from pwg.parallel_wavegan.utils import load_model
from pwg.parallel_wavegan.utils import read_hdf5

def audio_transform(wav_filepath, scaler, kwargs, device):

    trim_silence = kwargs['trim_silence']
    top_db = kwargs['top_db']
    flen = kwargs['flen']
    fshift = kwargs['fshift']
    fmin = kwargs['fmin']
    fmax = kwargs['fmax']
    num_mels = kwargs['num_mels']
    fs = kwargs['fs']

    #import pdb; pdb.set_trace()
    audio, fs_ = sf.read(wav_filepath)
    if trim_silence:
        #print('trimming.')
        audio, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=2048, hop_length=512)
    if fs != fs_:
        #print('resampling.')
        audio = librosa.resample(audio, fs_, fs)
    melspec_raw = logmelfilterbank(audio,fs, fft_size=flen,hop_size=fshift,
                                    fmin=fmin, fmax=fmax, num_mels=num_mels)
    melspec_raw = melspec_raw.astype(np.float32) # n_frame x n_mels

    melspec_norm = scaler.transform(melspec_raw)
    melspec_norm =  melspec_norm.T # n_mels x n_frame

    return torch.tensor(melspec_norm[None]).to(device, dtype=torch.float)

def make_onehot(clsidx, dim, device):
    onehot = np.eye(dim, dtype=np.int)[clsidx]
    return torch.tensor(onehot).to(device, dtype=torch.float)

def extract_num(s, p, ret=0):
    search = p.search(s)
    if search:
        return int(search.groups()[0])
    else:
        return ret

def listdir_ext(dirpath,ext):
    p = re.compile(r'(\d+)')
    out = []
    for file in sorted(os.listdir(dirpath), key=lambda s: extract_num(s, p)):
        if os.path.splitext(file)[1]==ext:
            out.append(file)
    return out

def find_newest_model_file(model_dir, tag):
    mfile_list = os.listdir(model_dir)
    checkpoint = max([int(os.path.splitext(os.path.splitext(mfile)[0])[0]) for mfile in mfile_list if mfile.endswith('.{}.pt'.format(tag))])
    return '{}.{}.pt'.format(checkpoint,tag)


def synthesis(melspec, pwg, pwg_config, savepath, device):
    ## Parallel WaveGAN / MelGAN
    #import pdb; pdb.set_trace()
    melspec = torch.tensor(melspec, dtype=torch.float).to(device)
    #start = time.time()
    x = pwg.inference(melspec).view(-1)
    #elapsed_time = time.time() - start
    #rtf2 = elapsed_time/audio_len
    #print ("elapsed_time (waveform generation): {0}".format(elapsed_time) + "[sec]")
    #print ("real time factor (waveform generation): {0}".format(rtf2))
    
    # save as PCM 16 bit wav file
    #import pdb;pdb.set_trace() # Breakpoint
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
    sf.write(savepath, x.detach().cpu().clone().numpy(), pwg_config["sampling_rate"], "PCM_16")

def main():
    parser = argparse.ArgumentParser(description='ACVAE-VC')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('-i', '--input', type=str, default='/misc/raid58/kameoka.hirokazu/python/db/arctic/wav/test',
                        help='root data folder that contains the wav files of input speech')
    parser.add_argument('-o', '--out', type=str, default='./out/arctic',
                        help='root data folder where the wav files of the converted speech will be saved.')
    parser.add_argument('--dataconf', type=str, default='./dump/arctic/data_config.json')
    parser.add_argument('--stat', type=str, default='./dump/arctic/stat.pkl', help='state file used for normalization')                        
    parser.add_argument('--model_rootdir', '-mdir', type=str, default='./model/arctic/', help='model file directory')
    parser.add_argument('--refine_type', '-ref', type=str, default='raw', help='attention refining process type')
    parser.add_argument('--experiment_name', '-exp', default='experiment1', type=str, help='experiment name')
    parser.add_argument('--vocoder', '-voc', default='parallel_wavegan.v1', type=str,
                        help='neural vocoder type name (e.g., parallel_wavegan.v1, melgan.v3.long)')
    parser.add_argument('--voc_dir', '-vdir', type=str, default='pwg/egs/arctic_4spk_flen64ms_fshift8ms/voc1', 
                        help='directory of trained neural vocoder')
    args = parser.parse_args()

    # Set up GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    input_dir = args.input
    data_config_path = args.dataconf
    model_config_path = os.path.join(args.model_rootdir,args.experiment_name,'model_config.json')
    with open(data_config_path) as f:
        data_config = json.load(f)
    with open(model_config_path) as f:
        model_config = json.load(f)

    num_mels = model_config['num_mels']
    n_spk = model_config['n_spk']
    trg_spk_list = model_config['spk_list']
    zdim = model_config['zdim']
    mdim = model_config['mdim']
    kdim = model_config['kdim']
    hdim = model_config['hdim']
    num_layers = model_config['num_layers']
    num_blocks = model_config['num_blocks']
    reduction_factor = model_config['reduction_factor']
    pos_weight = model_config['pos_weight']
    normtype = model_config['normtype']
    refine_type = args.refine_type

    stat_filepath = args.stat
    melspec_scaler = StandardScaler()
    if os.path.exists(stat_filepath):
        with open(stat_filepath, mode='rb') as f:
            melspec_scaler = pickle.load(f)
        print('Loaded mel-spectrogram statistics successfully.')
    else:
        print('Stat file not found.')

    # Set up main model
    models = {
        'enc_src' : net.SrcEncoder1(num_mels*reduction_factor,n_spk,hdim,zdim,kdim,num_layers,num_blocks,normtype),
        'enc_trg' : net.TrgEncoder1(num_mels*reduction_factor,n_spk,hdim,zdim,kdim,num_layers,num_blocks,normtype),
        'dec' : net.Decoder1(zdim*2,n_spk,hdim,num_mels*reduction_factor,mdim,num_layers,num_blocks,normtype)
    }
    models['convs2s'] = net.ConvS2S(models['enc_src'], models['enc_trg'], models['dec'])

    for tag in ['enc_src', 'enc_trg', 'dec']:
        model_dir = os.path.join(args.model_rootdir,args.experiment_name)
        mfilename = find_newest_model_file(model_dir, tag)
        #import pdb; pdb.set_trace()
        path = os.path.join(args.model_rootdir,args.experiment_name,mfilename)
        if path is not None:
            checkpoint = torch.load(path, map_location=device)
            models[tag].load_state_dict(checkpoint['model_state_dict'])
            print('{}: {}'.format(tag, mfilename))

    for tag in ['enc_src', 'enc_trg', 'dec']:
        models[tag].to(device).eval()

    # Set up PWG
    vocoder = args.vocoder
    voc_dir = args.voc_dir
    voc_yaml_path = os.path.join(voc_dir,'conf', '{}.yaml'.format(vocoder))
    checkpointlist = listdir_ext(
        os.path.join(voc_dir,'exp','train_nodev_all_{}'.format(vocoder)),'.pkl')
    pwg_checkpoint = os.path.join(voc_dir,'exp',
                                  'train_nodev_all_{}'.format(vocoder),
                                  checkpointlist[-1]) # Find and use the newest checkpoint model.
    print('vocoder type: {}'.format(vocoder))
    print('checkpoint  : {}'.format(checkpointlist[-1]))
    
    with open(voc_yaml_path) as f:
        pwg_config = yaml.load(f, Loader=yaml.Loader)
    pwg_config.update(vars(args))
    pwg = load_model(pwg_checkpoint, pwg_config)
    pwg.remove_weight_norm()
    pwg = pwg.eval().to(device)

    src_spk_list = sorted(os.listdir(input_dir))
    
    for i, src_spk in enumerate(src_spk_list):
        src_wav_dir = os.path.join(input_dir, src_spk)
        for j, trg_spk in enumerate(trg_spk_list):
            print('Converting {}2{}...'.format(src_spk, trg_spk))
            for n, src_wav_filename in enumerate(os.listdir(src_wav_dir)):
                src_wav_filepath = os.path.join(src_wav_dir, src_wav_filename)
                src_melspec = audio_transform(src_wav_filepath, melspec_scaler, data_config, device)

                conv_melspec, A, elapsed_time = models['convs2s'].inference(src_melspec, i, j, reduction_factor, 0.0, pos_weight, refine_type)
                conv_melspec = conv_melspec.T # n_frames x n_mels

                out_wavpath = os.path.join(args.out,args.experiment_name,'{}2{}'.format(src_spk,trg_spk), src_wav_filename)
                synthesis(conv_melspec, pwg, pwg_config, out_wavpath, device)


if __name__ == '__main__':
    main()