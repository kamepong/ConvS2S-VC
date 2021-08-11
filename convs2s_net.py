import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import scipy
from scipy import signal
import time

class DilCausConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, df, normtype='CBN'):
        super(DilCausConv1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, ks, dilation=df, padding=df*(ks-1))
        if normtype=='WN':
            self.conv1 = nn.utils.weight_norm(self.conv1)

    def __call__(self, x):
        xlen = x.shape[2]
        h = self.conv1(x)
        h = h[:,:,0:xlen]
        return h

class DilConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, df, normtype='CBN'):
        super(DilConv1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, ks, dilation=df, padding=df*(ks-1)//2)
        if normtype=='WN':
            self.conv1 = nn.utils.weight_norm(self.conv1)

    def __call__(self, x):
        h = self.conv1(x)
        return h

class DilCausConvGLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, df, clsnum, normtype='CBN'):
        super(DilCausConvGLU1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch*2, ks, dilation=df, padding=df*(ks-1))
        if normtype=='WN':
            self.conv1 = nn.utils.weight_norm(self.conv1)
        elif normtype=='CBN':
            self.norms1 = nn.ModuleList()
            for j in range(clsnum):
                self.norms1.append(nn.BatchNorm1d(out_ch*2))
        self.normtype = normtype

    def __call__(self, x, c):
        xlen = x.shape[2]
        h = x
        
        h = self.conv1(h)
        h = h[:,:,0:xlen]
        if self.normtype=='CBN':
            h = self.norms1[c](h)
                
        h_l, h_g = torch.split(h, h.shape[1]//2, dim=1)
        h = h_l * torch.sigmoid(h_g)     
        return h

class DilCausConvReLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, df, normtype='CBN'):
        super(DilCausConvReLU1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, ks,
            dilation=df, padding=df*(ks-1))
        if normtype=='WN':
            self.conv1 = nn.utils.weight_norm(self.conv1)
        
    def __call__(self, x):
        xlen = x.shape[2]
        h = self.conv1(x)
        h = h[:,:,0:xlen]
        h = F.softplus(h, beta=1.0)
        return h
    
class DilConvGLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, df, clsnum, normtype='CBN'):
        super(DilConvGLU1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch*2, ks, dilation=df, padding=df*(ks-1)//2)
        if normtype=='WN':
            self.conv1 = nn.utils.weight_norm(self.conv1)
        elif normtype=='CBN':
            self.norms1 = nn.ModuleList()
            for j in range(clsnum):
                self.norms1.append(nn.BatchNorm1d(out_ch*2))
        self.normtype = normtype

    def __call__(self, x, c):
        h = x
        
        h = self.conv1(h)
        if self.normtype=='CBN':
            h = self.norms1[c](h)
        
        h_l, h_g = torch.split(h, h.shape[1]//2, dim=1)
        h = h_l * torch.sigmoid(h_g)
        
        return h

class DilConvReLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, df, normtype='CBN'):
        super(DilConvReLU1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, ksize=ks,
            dilation=df, padding=df*(ks-1)//2)
        if normtype=='WN':
            self.conv1 = nn.utils.weight_norm(self.conv1)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.softplus(h, beta=1.0)
        return h
    
def position_encoding(length, n_units):
    '''
    # Implementation described in the paper
    start = 1  # index starts from 1 or 0
    posi_block = np.arange(
        start, length + start, dtype='f')[None, None, :]
    unit_block = np.arange(
        start, n_units // 2 + start, dtype='f')[None, :, None]
    rad_block = posi_block / 10000. ** (unit_block / (n_units // 2))
    sin_block = np.sin(rad_block)
    cos_block = np.cos(rad_block)
    position_encoding_block = np.empty((1, n_units, length), 'f')
    position_encoding_block[:, ::2, :] = sin_block
    position_encoding_block[:, 1::2, :] = cos_block
    ''' 

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

def xy_concat(x,y):
    N, aux_ch = y.shape
    y0 = torch.unsqueeze(y,2)
    N, n_ch, n_q = x.shape
    yy = y0.repeat(1,1,n_q)
    h = torch.cat((x,yy), dim=1)
    return h

class SrcEncoder1(nn.Module):
    # 1D Dilated Non-Causal Convolution
    def __init__(self, in_ch, clsnum, h_ch, out_ch, mid_ch, num_layers, num_blocks, normtype='CBN'):
        super(SrcEncoder1, self).__init__()
        
        self.layer_names = []
        assert num_layers > 1
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.clsnum = clsnum
        
        self.eb1 = nn.Embedding(clsnum, h_ch)
        #self.eb1 = nn.utils.weight_norm(self.eb1)
        self.start1 = DilConv1D(in_ch+h_ch,mid_ch,1,1,normtype)

        self.glu_blocks = nn.ModuleList()
        for i in range(num_layers*num_blocks):
            dilation = 3**(i%num_layers)
            self.glu_blocks.append(DilConvGLU1D(mid_ch+h_ch,mid_ch,5,dilation,clsnum,normtype))

        self.end1 = DilConv1D(mid_ch+h_ch,out_ch*2,1,1,normtype)
            
    def __call__(self, x, c, dor=0.05):
        device = x.device
        N, n_ch, n_t = x.shape
        t = torch.LongTensor(c*np.ones(N)).to(device, dtype=torch.int64)
        l = self.eb1(t)
        # l.shape: (N, h_ch)
        
        out = F.dropout(x, p=dor)
        out = xy_concat(out,l)
        out = self.start1(out)
        for i, layer in enumerate(self.glu_blocks):
            outl = xy_concat(out,l)
            out = layer(outl,c) + out
        out = xy_concat(out,l)
        out = self.end1(out)
        K, V = torch.split(out, out.shape[1]//2, dim=1)
        return K, V

class TrgEncoder1(nn.Module):
    # 1D Dilated Causal Convolution
    def __init__(self, in_ch, clsnum, h_ch, out_ch, mid_ch, num_layers, num_blocks, normtype='CBN'):
        super(TrgEncoder1, self).__init__()
        
        self.layer_names = []
        assert num_layers > 1
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.clsnum = clsnum
        
        self.eb1 = nn.Embedding(clsnum, h_ch)
        #self.eb1 = nn.utils.weight_norm(self.eb1)
        self.start1 = DilCausConv1D(in_ch+h_ch,mid_ch,1,1,normtype)

        self.glu_blocks = nn.ModuleList()
        for i in range(num_layers*num_blocks):
            dilation = 3**(i%num_layers)
            self.glu_blocks.append(DilCausConvGLU1D(mid_ch+h_ch,mid_ch,3,dilation,clsnum,normtype))

        self.end1 = DilCausConv1D(mid_ch+h_ch,out_ch,1,1,normtype)
            
    def __call__(self, x, c, dor=0.05):
        device = x.device
        N, n_ch, n_t = x.shape
        t = torch.LongTensor(c*np.ones(N)).to(device, dtype=torch.int64)
        l = self.eb1(t)
        # l.shape: (N, h_ch)
        
        out = F.dropout(x, p=dor)
        out = xy_concat(out,l)
        out = self.start1(out)

        for i, layer in enumerate(self.glu_blocks):
            outl = xy_concat(out,l)
            out = layer(outl,c) + out
        out = xy_concat(out,l)
        Q = self.end1(out)
        return Q

class Decoder1(nn.Module):
    # 1D Dilated Causal Convolution
    def __init__(self, in_ch, clsnum, h_ch, out_ch, mid_ch, num_layers, num_blocks, normtype='CBN'):
        super(Decoder1, self).__init__()
        
        self.layer_names = []
        assert num_layers > 1
        self.num_layers = num_layers
        self.num_blocks = num_blocks

        self.eb1 = nn.Embedding(clsnum, h_ch)
        #self.eb1 = nn.utils.weight_norm(self.eb1)
        self.start1 = DilCausConv1D(in_ch+h_ch,mid_ch,1,1,normtype)

        self.glu_blocks = nn.ModuleList()
        for i in range(num_layers*num_blocks):
            dilation = 3**(i%num_layers)
            self.glu_blocks.append(DilCausConvGLU1D(mid_ch+h_ch,mid_ch,3,dilation,clsnum,normtype))

        self.end1 = DilCausConv1D(mid_ch+h_ch,out_ch,1,1,normtype)

    def __call__(self, x, c, rf, dor=0.05):
        device = x.device
        N, n_ch, n_t = x.shape
        t = torch.LongTensor(c*np.ones(N)).to(device, dtype=torch.int64)
        l = self.eb1(t)
        # l.shape: (N, h_ch)
        
        out = F.dropout(x, p=dor)

        out = xy_concat(out,l)
        out = self.start1(out)

        for i, layer in enumerate(self.glu_blocks):
            outl = xy_concat(out,l)
            out = layer(outl,c) + out
        out = xy_concat(out,l)
        y = self.end1(out)

        return y

def gaussdis(N,mu,sigma):
    nN = np.arange(0,N)
    nN = nN.reshape(N,1)
    x = np.exp(-np.square(nN - mu)/(2.0*sigma**2))
    x = x/np.sum(x,keepdims=True)
    return x

def localpeak(A,mu,sigma):
    epsi = sys.float_info.epsilon
    A = A.flatten()
    N = len(A)
    gw = gaussdis(N,mu,sigma).flatten()
    wA = (A+epsi)*gw
    wA = wA/np.sum(wA)
    return np.argmax(wA), wA

def subsample(x, rf):
    device = x.device
    B,D,N = x.shape
    N_mod = math.ceil(N/rf)*rf
    if N_mod != N:
        z = torch.tensor(np.zeros((B, D, N_mod-N))).to(device, dtype=torch.float)
        x = torch.cat((x, z), dim=2)
    out = x.permute(0,2,1).reshape(B,N_mod//rf,D*rf).permute(0,2,1)
    return out

def expand(x, rf):
    device = x.device
    B,D,N = x.shape
    out = x.permute(0,2,1).reshape(B,N*rf,D//rf).permute(0,2,1)
    return out

def pad_zero_frame(x):
    device = x.device
    B,D,N = x.shape
    zero = torch.tensor(np.zeros((B,D,1))).to(device, dtype=torch.float)
    out = torch.cat((zero,x),dim=2)
    return out

class ConvS2S():
    def __init__(self, enc_src, enc_trg, dec):
        self.enc_src = enc_src
        self.enc_trg = enc_trg
        self.dec = dec

    def calc_loss(self, x_s, x_t, m_s, m_t, l_s, l_t, dor=0.1, pos_weight=1.0,
                          gauss_width_da=0.3, reduction_factor = 3):
        # L1 loss with position encoding
        device = x_s.device
        rf = reduction_factor
        # x_s.shape: batchsize x num_mels x N
        # x_t.shape: batchsize x num_mels x T
        #N = x_s.shape[2]
        #T = x_t.shape[2]
        num_mels = x_s.shape[1]
        BatchSize = x_s.shape[0]

        x_s = subsample(x_s,rf)
        x_t = subsample(x_t,rf)

        # Pad all-zero frame
        x_t = pad_zero_frame(x_t)
        
        B,D,N = x_s.shape
        B,D,T = x_t.shape
        assert D == num_mels*rf
        
        pos_s = position_encoding(N, D)
        pos_t = position_encoding(T, D)
        pos_s = torch.tensor(pos_s).to(device, dtype=torch.float)
        pos_t = torch.tensor(pos_t).to(device, dtype=torch.float)
        pos_s = pos_s.repeat(BatchSize,1,1)
        pos_t = pos_t.repeat(BatchSize,1,1)
        scale_emb = D**0.5

        in_s = x_s
        in_s[:,0:pos_s.shape[1],:] = in_s[:,0:pos_s.shape[1],:] + pos_s/scale_emb * pos_weight
        in_t = x_t
        in_t[:,0:pos_t.shape[1],:] = in_t[:,0:pos_t.shape[1],:] + pos_t/scale_emb * pos_weight
        
        m_s = m_s[:,:,0::rf]
        m_t = m_t[:,:,0::rf]
        zero = torch.tensor(np.zeros((BatchSize,1,1))).to(device, dtype=torch.float)
        m_t = torch.cat((zero,m_t),dim=2)
        assert m_s.shape[2] == N
        assert m_t.shape[2] == T
        
        #import pdb;pdb.set_trace() # Breakpoint
        
        K_s, V_s = self.enc_src(in_s, l_s, dor)

        # K_s.shape: 1 x d x N
        d = K_s.shape[1]
        Q_t = self.enc_trg(in_t, l_t, dor)
        # Q_t.shape: 1 x d x T

        # Attention matrix
        # Scaled dot-product attention
        A = F.softmax(torch.matmul(K_s.permute(0,2,1), Q_t)/math.sqrt(d), dim=1)

        # A.shape: 1 x N x T
        R = torch.matmul(V_s,A)
        # R.shape: 1 x d x T

        R = torch.cat((R,F.dropout(Q_t, p=0.9)), dim=1)

        y = self.dec(R,l_t,rf,dor)
        #import pdb;pdb.set_trace() # Breakpoint

        # Main Loss
        MainLoss = torch.sum(torch.mean(
            m_t[:,:,1:T].repeat(1,num_mels*rf,1)*
            torch.abs(y[:,:,0:T-1] - x_t[:,:,1:T]), 1))
        MainLoss = MainLoss/torch.sum(m_t[:,:,1:T])

        W = np.zeros((BatchSize,N,T))
        # Compute Penalty Matrix
        for b in range(0,BatchSize):
            Nb = int(torch.sum(m_s[b,:,:]))
            Tb = int(torch.sum(m_t[b,:,:]))
            nN = np.arange(0,N)/Nb
            tT = np.arange(0,T)/Tb
            nN_tiled = np.tile(nN[:,np.newaxis], (1,T))
            tT_tiled = np.tile(tT[np.newaxis,:], (N,1))
            W[b,:,:] = 1.0-np.exp(-np.square(nN_tiled - tT_tiled)/(2.0*gauss_width_da**2))
            W[b,Nb:N,Tb:T] = 0.
        W = torch.tensor(W).to(device, dtype=torch.float)
        
        # Diagonal Attention Loss
        DALoss = torch.sum(torch.mean(A*W, 1))/torch.sum(m_t)
        #import pdb;pdb.set_trace() # Breakpoint

        A_np = A.detach().cpu().clone().numpy()
        
        return MainLoss, DALoss, A_np

    def inference(self, x_s, l_s, l_t, rf, 
                  dor=0.0, pos_weight=1.0, refine='raw'):
        start = time.time()
        
        device = x_s.device
        # x_s.shape: batchsize x num_mels x N
        num_mels = x_s.shape[1]

        x_s = subsample(x_s, rf)
        BatchSize,D,N = x_s.shape

        pos_s = position_encoding(N, D)
        pos_s = torch.tensor(pos_s).to(device, dtype=torch.float)
        pos_s = pos_s.repeat(BatchSize,1,1)
        scale_emb = D**0.5

        in_s = x_s
        in_s[:,0:pos_s.shape[1],:] = in_s[:,0:pos_s.shape[1],:] + pos_s/scale_emb * pos_weight
        x_t = torch.tensor(np.zeros((1,D,1))).to(device, dtype=torch.float)

        self.enc_src.eval()
        self.enc_trg.eval()
        self.dec.eval()

        with torch.no_grad():
            K, V = self.enc_src(in_s,l_s,dor)
        d = K.shape[1]
        
        if refine == 'raw' or refine == None:
            # Raw attention
            T = round(N*2.0)
            for t in range(0,T):

                pos_t = position_encoding(x_t.shape[2], D)
                pos_t = torch.tensor(pos_t).to(device, dtype=torch.float)

                in_t = x_t
                in_t[:,0:pos_t.shape[1],:] = in_t[:,0:pos_t.shape[1],:] + pos_t/scale_emb * pos_weight
                
                with torch.no_grad():
                    Q = self.enc_trg(in_t, l_t, dor)
                    # Scaled dot-product attention
                    A = F.softmax(torch.matmul(K.permute(0,2,1), Q)/math.sqrt(d), dim=1)
                    R = torch.matmul(V,A)
                    R = torch.cat((R,F.dropout(Q, p=0.0, training=False)), dim=1)
                    y = self.dec(R,l_t,rf,dor)
                    Zero = np.zeros((1,D,1))
                    zero = torch.tensor(Zero).to(device, dtype=torch.float)
                    x_t = torch.cat((zero,y), dim=2)

            elapsed_time = time.time() - start
            amod = A.clone()
            A_np = A[0,:,:].detach().cpu().clone().numpy()**0.3
            path = mydtw_fromDistMat(1.0-A_np,w=100,p=0.1)

            end_of_frame = path[1][-1]
            #end_of_frame = min(path[1][-1]+20, T)
                
        elif refine == 'diagonal':
            # Exactly diagonal attention (no time-warping)
            T = N
            end_of_frame = T

            for t in range(0,T):
                pos_t = position_encoding(x_t.shape[2], D)
                pos_t = torch.tensor(pos_t).to(device, dtype=torch.float)

                in_t = x_t
                in_t[:,0:pos_t.shape[1],:] = in_t[:,0:pos_t.shape[1],:] + pos_t/scale_emb * pos_weight

                with torch.no_grad():
                    Q = self.enc_trg(in_t, l_t, dor)
                    R = torch.cat((V[:,:,0:t+1],F.dropout(Q, p=0.0, training=False)), dim=1)
                    y = self.dec(R,l_t,rf,dor)
                    Zero = np.zeros((1,D,1))
                    zero = torch.tensor(Zero).to(device, dtype=torch.float)
                    x_t = torch.cat((zero,y), dim=2)

            elapsed_time = time.time() - start
            Amod = np.eye(N).reshape(1,N,N)
            amod = torch.tensor(Amod).to(device, dtype=torch.float)
            path = [np.arange(N), np.arange(N)]

        if refine == 'forward':
            # Forward attention
            T = round(N*2.0)
            n_argmax = 0
            y_samples = np.array([0])
            x_samples = np.array([0])
            for t in range(0,T):

                pos_t = position_encoding(x_t.shape[2], D)
                pos_t = torch.tensor(pos_t).to(device, dtype=torch.float)

                in_t = x_t
                in_t[:,0:pos_t.shape[1],:] = in_t[:,0:pos_t.shape[1],:] + pos_t/scale_emb * pos_weight

                with torch.no_grad():
                    Q = self.enc_trg(in_t, l_t, dor)
                    # Scaled dot-product attention
                    A = F.softmax(torch.matmul(K.permute(0,2,1), Q)/math.sqrt(d), dim=1)
                    
                    # Scaled dot-product attention
                    A = F.softmax(torch.matmul(K.permute(0,2,1), Q)/math.sqrt(d), dim=1)
                    if t == 0:
                        A_concat = A.clone()
                    else:
                        A_last = A[:,:,t:t+1].detach().cpu().clone().numpy()
                        A_last[0,0:max(n_argmax-20//rf,0),0] = 0
                        A_last[0,min(n_argmax+40//rf,N-1):,0] = 0
                        A_last = (np.maximum(A_last,1e-10))/np.sum(np.maximum(A_last,1e-10))
                        a_last = torch.tensor(A_last).to(device, dtype=torch.float)
                        A_concat = torch.cat((A_concat, a_last), dim=2)
                    
                    amod = A_concat.clone()
                    A = A_concat.clone()
                    A_np = A[0,:,t].detach().cpu().clone().numpy()

                    # Prediction of attended time point
                    if t==0:
                        n_argmax_tmp = localpeak(A_np,n_argmax,5.0)[0]
                        if n_argmax_tmp > n_argmax:
                            n_argmax = n_argmax_tmp
                    else:
                        n_argmax_tmp = localpeak(A_np,n_argmax,5.0)[0]
                        y_samples = np.append(y_samples,n_argmax_tmp)
                        x_samples = np.append(x_samples,t)
                        slope = (np.mean((y_samples-np.mean(y_samples))*(x_samples-np.mean(x_samples)))
                                 /(max(np.std(x_samples),1e-10)**2))
                        n_argmax = int(round(slope*(t+1)))

                    #import pdb;pdb.set_trace() # Breakpoint

                    R = torch.matmul(V,A)
                    R = torch.cat((R,F.dropout(Q, p=0.0, training=False)), dim=1)
                    y = self.dec(R,l_t,rf,dor)
                    
                    Zero = np.zeros((1,D,1))
                    zero = torch.tensor(Zero).to(device, dtype=torch.float)
                    x_t = torch.cat((zero,y), dim=2)

            elapsed_time = time.time() - start
            A_tmp = A[0,:,:].detach().cpu().clone().numpy()**0.3
            #import pdb;pdb.set_trace() # Breakpoint
            path = mydtw_fromDistMat(1.0-A_tmp,w=100,p=0.1)
            end_of_frame = path[1][-1]
            #end_of_frame = T
            
        A = amod[:,:,0:end_of_frame].clone()
        A_out = A.detach().cpu().clone().numpy()
        #import pdb;pdb.set_trace() # Breakpoint
        with torch.no_grad():
            R = torch.matmul(V,A)
            R = torch.cat((R,F.dropout(Q[:,:,0:end_of_frame], p=0.0, training=False)), dim=1)
            y = self.dec(R,l_t,rf,dor)

        #melspec_conv = expand(y[:,0:D,:],rf).detach().cpu().clone().numpy()
        melspec_conv = expand(y[:,0:D,0:end_of_frame],rf).detach().cpu().clone().numpy()
        melspec_conv = melspec_conv[0,:,:]

        #import pdb;pdb.set_trace() # Breakpoint
        return melspec_conv, A_out, elapsed_time

def mydtw_fromDistMat(D0, w=np.inf, p=0.0):
    r, c = D0.shape
    AccDis = np.full(D0.shape, np.inf)
    AccDis[0,0] = D0[0,0]
    pointer = np.full(D0.shape, 0)
    irange = range(1,min(r,1+w+1))
    for i in irange:
        AccDis[i,0] = AccDis[i-1,0] + p + D0[i,0]
        pointer[i,0] = 1 #means "came from down"
    jrange = range(1,min(c,1+w+1))
    for j in jrange:
        AccDis[0,j] = AccDis[0,j-1] + p + D0[0,j]
        pointer[0,j] = 2 #means "came from left"
    
    for i in range(1,r):
        jrange = range(max(1,i-w),min(c,i+w+1))
        for j in jrange:
            AccDis[i,j] = np.min([AccDis[i-1,j-1], AccDis[i-1,j]+p, AccDis[i,j-1]+p]) + D0[i,j]
            pointer[i,j] = np.argmin([AccDis[i-1,j-1], AccDis[i-1,j]+p, AccDis[i,j-1]+p])
            
    if np.min(AccDis[:,c-1])<np.min(AccDis[r-1,:]):
        r_end = np.argmin(AccDis[:,c-1])
        c_end = c-1
    else:
        r_end = r-1
        c_end = np.argmin(AccDis[r-1,:])

    #import pdb;pdb.set_trace() # Breakpoint
        
    # trace back
    path_r, path_c = [r_end], [c_end]
    i, j = r_end, c_end
    while (i > 0) or (j > 0):
        if pointer[i,j]==0:
            i -= 1
            j -= 1
        elif pointer[i,j]==1:
            i -= 1
        else: #pointer[i,j]==2:
            j -= 1
        path_r.insert(0, i)
        path_c.insert(0, j)

    #import pdb;pdb.set_trace() # Breakpoint
    return np.array(path_r), np.array(path_c)