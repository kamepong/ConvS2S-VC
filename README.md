# ConvS2S-VC

This repository provides a PyTorch implementation for [ConvS2S-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/convs2s-vc2/index.html).

ConvS2S-VC uses a fully convolutional sequence-to-sequence (ConvS2S) model to learn mappings between the mel-spectrograms of source and target speech, and [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) to generate a waveform from the converted mel-spectrogram. 

Audio samples are available [here](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/convs2s-vc2/index.html).

#### Prerequisites

- PyTorch>=1.2.0
- See https://github.com/kan-bayashi/ParallelWaveGAN for the packages needed to set up Parallel WaveGAN.



## Paper

**FastS2S-VC: Streaming Non-Autoregressive Sequence-to-Sequence Voice Conversion**
[Hirokazu Kameoka](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/index-e.html), [Kou Tanaka](http://www.kecl.ntt.co.jp/people/tanaka.ko/index.html), [Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/index.html)
arXiv:2104.06900 [cs.SD], 2021.

**ConvS2S-VC: Fully Convolutional Sequence-to-Sequence Voice Conversion**
[Hirokazu Kameoka](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/index-e.html), [Kou Tanaka](http://www.kecl.ntt.co.jp/people/tanaka.ko/index.html), Damian Kwasny, [Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/index.html), Nobukatsu Hojo
*IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 28, pp. 1849-1863, Jun. 2020.
[**[Paper]**](https://ieeexplore.ieee.org/document/9113442) 



## Preparation

#### Dataset

1. Setup your training dataset. The data structure should look like:

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
```

#### Parallel WaveGAN

1. Setup Parallel WaveGAN.  Place a copy of the directory `parallel_wavegan` from https://github.com/kan-bayashi/ParallelWaveGAN in `pwg/`.
2. Parallel WaveGAN models trained on several databases can be found [here](https://app.box.com/folder/127558077224). Once these are downloaded, place them in `pwg/egs/`. Please contact me if you have any problems downloading.

```bash
# Model trained on the ATR database (11 speakers)
cp -r ATR_all_flen64ms_fshift8ms pwg/egs/
# Model trained on the CMU ARCTIC dataset (4 speakers)
cp -r arctic_4spk_flen64ms_fshift8ms pwg/egs/
```



## Main

See shell scripts in `recipes` for examples of training on different datasets.

#### Feature Extraction

To extract the normalized mel-spectrograms from the training utterances, execute:

```bash
python extract_features.py
python compute_statistics.py
python normalize_features.py
```

#### Train

To train the model, execute:

```bash
python main.py -g 0
```

#### Test

To perform conversion, execute:

```bash
python convert.py -g 0
```



## Citation

If you find this work useful for your research, please cite our paper.

```
@Article{Kameoka2020IEEETrans_ConvS2S-VC,
  author={Hirokazu Kameoka and Kou Tanaka and Damian Kwasny and Takuhiro Kaneko and Nobukatsu Hojo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={ConvS2S-VC: Fully Convolutional Sequence-to-Sequence Voice Conversion}, 
  year={2020},
  month=jun,
  volume={28},
  number={},
  pages={1849-1863}
}
```



## Author

Hirokazu Kameoka ([@kamepong](https://github.com/kamepong))

E-mail: kame.hirokazu@gmail.com
