# ConvS2S-VC

This repository provides an official PyTorch implementation for [ConvS2S-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/s2s-vc/index.html).

ConvS2S-VC is a parallel many-to-many voice conversion (VC) method using a fully convolutional sequence-to-sequence (ConvS2S) model. The current version performs VC by first modifying the mel-spectrogram of input speech in accordance with a target speaker or style index, and then generating a waveform using a speaker-independent neural vocoder (Parallel WaveGAN or HiFi-GAN) from the converted mel-spectrogram. 

Audio samples are available [here](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/s2s-vc/index.html).

## Papers

- [Hirokazu Kameoka](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/index-e.html), [Kou Tanaka](http://www.kecl.ntt.co.jp/people/tanaka.ko/index.html), [Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/index.html), "**FastS2S-VC: Streaming Non-Autoregressive Sequence-to-Sequence Voice Conversion**," arXiv:2104.06900 [cs.SD], 2021. [**[Paper]**](https://arxiv.org/abs/2104.06900) 

- [Hirokazu Kameoka](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/index-e.html), [Kou Tanaka](http://www.kecl.ntt.co.jp/people/tanaka.ko/index.html), Damian Kwasny, [Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/index.html), Nobukatsu Hojo, "**ConvS2S-VC: Fully Convolutional Sequence-to-Sequence Voice Conversion**," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 28, pp. 1849-1863, Jun. 2020. [**[Paper]**](https://ieeexplore.ieee.org/document/9113442) 

## Preparation

#### Requirements

- See `requirements.txt`.

#### Dataset

1. Setup your training and test sets. The data structure should look like:

```bash
/path/to/dataset/training
├── spk_1
│   ├── utt1.wav
│   ...
├── spk_2
│   ├── utt1.wav
│   ...
└── spk_N
    ├── utt1.wav
    ...
    
/path/to/dataset/test
├── spk_1
│   ├── utt1.wav
│   ...
├── spk_2
│   ├── utt1.wav
│   ...
└── spk_N
    ├── utt1.wav
    ...
```

#### Waveform generator

1. Place a copy of the directory `parallel_wavegan` from https://github.com/kan-bayashi/ParallelWaveGAN in `pwg/`.
2. HifiGAN models trained on several databases can be found [here](https://drive.google.com/drive/folders/1RvagKsKaCih0qhRP6XkSF07r3uNFhB5T?usp=sharing). Once these are downloaded, place them in `pwg/egs/`. Please contact me if you have any problems downloading.
3. Optionally, Parallel WaveGAN can be used instead for waveform generation. The trained models are available [here](https://drive.google.com/drive/folders/1zRYZ9dx16dONn1SEuO4wXjjgJHaYSKwb?usp=sharing). Once these are downloaded, place them in `pwg/egs/`. 

## Main

#### Train

To run all stages for model training, execute:

```bash
./recipes/run_train.sh [-g gpu] [-s stage] [-e exp_name]
```

- Options:

  ```bash
  -g: GPU device (default: -1)
  #    -1 indicates CPU
  -s: Stage to start (0 or 1)
  #    Stages 0 and 1 correspond to feature extraction and model training, respectively.
  -e: Experiment name (default: "exp1")
  #    This name will be used at test time to specify which trained model to load.
  ```

- Examples:

  ```bash
  # To run the training from scratch with the default settings:
  ./recipes/run_train.sh
  
  # To skip the feature extraction stage:
  ./recipes/run_train.sh -s 1
  
  # To set the gpu device to, say, 0:
  ./recipes/run_train.sh -g 0
  ```

See other scripts in `recipes` for examples of training on different datasets. 

To monitor the training process, use tensorboard:

```bash
tensorboard [--logdir log_path]
```

#### Test

To perform conversion, execute:

```bash
./recipes/run_test.sh [-g gpu] [-e exp_name] [-c checkpoint] [-a attention_mode] [-v vocoder]
```

- Options:

  ```bash
  -g: GPU device (default: -1)
  #    -1 indicates CPU
  -e: Experiment name (e.g., "exp1")
  -c: Model checkpoint to load (default: 0)
  #    0 indicates the newest model
  -a: Attention mode ("r", "f", or "d")
  #    The modes in which the attention matrix is processed during conversion
  #    r: raw attention (default)
  #    f: windowed attention
  #    d: exactly diagonal attention
  -v: Vocoder type ("hfg" or "pwg")
  #    The type of vocoder used for waveform generation
  #    hfg: HiFi-GAN (default)
  #    pwg: Parallel WaveGAN
  ```

- Examples:

  ```bash
  # To perform conversion with the default settings:
  ./recipes/run_test.sh -g 0 -e exp1
  
  # To enable attention windowing:
  ./recipes/run_test.sh -g 0 -e exp1 -a f
  
  # To use Parallel WaveGAN as an alternative for waveform generation:
  ./recipes/run_test.sh -g 0 -e exp1 -v pwg
  ```

## Citation

If you find this work useful for your research, please cite our papers.

```
@Article{Kameoka2021arXiv_FastS2S-VC,
  author={Hirokazu Kameoka and Kou Tanaka and Takuhiro Kaneko},
  journal={arXiv:2104.06900 [cs.SD]}, 
  title={FastS2S-VC: Streaming Non-Autoregressive Sequence-to-Sequence Voice Conversion}, 
  year={2021},
  month=apr
}
@Article{Kameoka2020IEEETrans_ConvS2S-VC,
  author={Hirokazu Kameoka and Kou Tanaka and Damian Kwasny and Takuhiro Kaneko and Nobukatsu Hojo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={{ConvS2S-VC}: Fully Convolutional Sequence-to-Sequence Voice Conversion}, 
  year={2020},
  month=jun,
  volume={28},
  pages={1849-1863}
}
```



## Author

Hirokazu Kameoka ([@kamepong](https://github.com/kamepong))

E-mail: kame.hirokazu@gmail.com
