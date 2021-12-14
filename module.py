# Copyright 2021 Hirokazu Kameoka

import numpy as np
import torch
import torch.nn as nn

def calc_padding(kernel_size, dilation, causal, stride=1):
    if causal:
        padding = (kernel_size-1)*dilation + 1 - stride
    else:
        padding = ((kernel_size-1)*dilation + 1 - stride)//2
    return padding

class DilCausConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, df):
        super(DilCausConv1D, self).__init__()
        self.padding = calc_padding(ks, df, True)
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, ks, dilation=df, padding=0)
        self.conv1 = nn.utils.weight_norm(self.conv1)

    def forward(self, input, state=None):
        if state is None:
            state = torch.zeros_like(input[:, :, :1]).repeat(1, 1, self.padding)
        input = torch.cat([state, input], dim=2)
        output = self.conv1(input)
        state = input[:, :, -self.padding:]
        
        return output, state

class DilConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, df):
        super(DilConv1D, self).__init__()
        self.padding = calc_padding(ks, df, False)
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, ks, dilation=df, padding=self.padding)
        self.conv1 = nn.utils.weight_norm(self.conv1)

    def forward(self, x):
        h = self.conv1(x)
        return h

class DilCausConvGLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, df):
        super(DilCausConvGLU1D, self).__init__()
        self.padding = calc_padding(ks, df, True)
        self.conv1 = nn.Conv1d(
            in_ch, out_ch*2, ks, dilation=df, padding=0)
        self.conv1 = nn.utils.weight_norm(self.conv1)

    def forward(self, input, state=None):
        if state is None:
            state = torch.zeros_like(input[:, :, :1]).repeat(1, 1, self.padding)
        input = torch.cat([state, input], dim=2)
        output = self.conv1(input)
        state = input[:, :, -self.padding:]
                
        h_l, h_g = torch.split(output, output.shape[1]//2, dim=1)
        output = h_l * torch.sigmoid(h_g)     
        return output, state

class DilConvGLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, df):
        super(DilConvGLU1D, self).__init__()
        self.padding = calc_padding(ks, df, False)
        self.conv1 = nn.Conv1d(
            in_ch, out_ch*2, ks, dilation=df, padding=self.padding)
        self.conv1 = nn.utils.weight_norm(self.conv1)

    def forward(self, x):
        h = x        
        h = self.conv1(h)
        h_l, h_g = torch.split(h, h.shape[1]//2, dim=1)
        h = h_l * torch.sigmoid(h_g)
        
        return h
   
def position_encoding(length, n_units):
    # Implementation in the Google tensor2tensor repo
    channels = n_units
    position = np.arange(length, dtype='f')
    num_timescales = channels // 2
    log_timescale_increment = (
        np.log(10000. / 1.) /
        (float(num_timescales) - 1))
    inv_timescales = 1. * np.exp(
        np.arange(num_timescales).astype('f') * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate(
        [np.sin(scaled_time), np.cos(scaled_time)], axis=1)

    #import pdb;pdb.set_trace() # Breakpoint
    signal = np.expand_dims(signal,axis=0)
    #signal = np.reshape(signal, [1, length, channels])
    position_encoding_block = np.transpose(signal, (0, 2, 1))
    return position_encoding_block

def concat_dim1(x,y):
    num_batch, aux_ch = y.shape
    y0 = torch.unsqueeze(y,2)
    num_batch, n_ch, n_t = x.shape
    yy = y0.repeat(1,1,n_t)
    h = torch.cat((x,yy), dim=1)
    return h