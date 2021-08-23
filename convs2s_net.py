# Copyright 2021 Hirokazu Kameoka

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import module as md

class SrcEncoder1(nn.Module):
    # 1D Dilated Non-Causal Convolution
    def __init__(self, in_ch, clsnum, h_ch, out_ch, mid_ch, num_layers=8, dor=0.1):
        super(SrcEncoder1, self).__init__()
        
        self.layer_names = []
        assert num_layers > 1
        self.num_layers = num_layers
        self.clsnum = clsnum
        
        self.eb = nn.Embedding(clsnum, h_ch)
        #self.eb = nn.utils.weight_norm(self.eb)
        self.start = md.DilConv1D(in_ch+h_ch,mid_ch,1,1)

        dilation = [3**(i%4) for i in range(num_layers)]
        # [1, 3, 9, 27, 1, 3, 9, 27]
        self.glu_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.glu_blocks.append(md.DilConvGLU1D(mid_ch+h_ch,mid_ch,5,dilation[i]))

        self.end = md.DilConv1D(mid_ch+h_ch,out_ch*2,1,1)
        self.dropout = nn.Dropout(p=dor)
        
    def __call__(self, x, c):
        device = x.device
        N, n_ch, n_t = x.shape
        t = torch.LongTensor(c*np.ones(N)).to(device, dtype=torch.int64)
        l = self.eb(t)
        # l.shape: (N, h_ch)
        
        out = self.dropout(x)
        out = md.concat_dim1(out,l)
        out = self.start(out)
        for i, layer in enumerate(self.glu_blocks):
            outl = md.concat_dim1(out,l)
            out = layer(outl) + out
        out = md.concat_dim1(out,l)
        out = self.end(out)
        K, V = torch.split(out, out.shape[1]//2, dim=1)
        return K, V

class TrgEncoder1(nn.Module):
    # 1D Dilated Causal Convolution
    def __init__(self, in_ch, clsnum, h_ch, out_ch, mid_ch, num_layers=8, dor=0.1):
        super(TrgEncoder1, self).__init__()
        
        self.layer_names = []
        assert num_layers > 1
        self.num_layers = num_layers
        self.clsnum = clsnum
        
        self.eb = nn.Embedding(clsnum, h_ch)
        #self.eb = nn.utils.weight_norm(self.eb)
        self.start = md.DilConv1D(in_ch+h_ch,mid_ch,1,1)

        dilation = [3**(i%4) for i in range(num_layers)]
        # [1, 3, 9, 27, 1, 3, 9, 27]
        self.glu_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.glu_blocks.append(md.DilCausConvGLU1D(mid_ch+h_ch,mid_ch,5,dilation[i]))

        self.end = md.DilConv1D(mid_ch+h_ch,out_ch,1,1)
        self.dropout = nn.Dropout(p=dor)
            
    def __call__(self, x, c, state=None):
        if state is None:
            state = [None]*self.num_layers
        device = x.device
        N, n_ch, n_t = x.shape
        t = torch.LongTensor(c*np.ones(N)).to(device, dtype=torch.int64)
        l = self.eb(t)
        # l.shape: (N, h_ch)
        
        out = self.dropout(x)
        out = md.concat_dim1(out,l)
        out = self.start(out)

        for i, layer in enumerate(self.glu_blocks):
            outl = md.concat_dim1(out,l)
            _out, _state = layer(outl,state=state.pop(0))
            state += [_state]
            out = _out + out
        out = md.concat_dim1(out,l)
        Q = self.end(out)
        return Q, state

class Decoder1(nn.Module):
    # 1D Dilated Causal Convolution
    def __init__(self, in_ch, clsnum, h_ch, out_ch, mid_ch, num_layers=8, dor=0.1):
        super(Decoder1, self).__init__()
        
        self.layer_names = []
        assert num_layers > 1
        self.num_layers = num_layers

        self.eb = nn.Embedding(clsnum, h_ch)
        #self.eb = nn.utils.weight_norm(self.eb)
        self.start = md.DilConv1D(in_ch+h_ch,mid_ch,1,1)

        dilation = [3**(i%4) for i in range(num_layers)]
        # [1, 3, 9, 27, 1, 3, 9, 27]
        self.glu_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.glu_blocks.append(md.DilCausConvGLU1D(mid_ch+h_ch,mid_ch,3,dilation[i]))

        self.end = md.DilConv1D(mid_ch+h_ch,out_ch,1,1)
        self.dropout = nn.Dropout(p=dor)

    def __call__(self, x, c, state=None):
        if state is None:
            state = [None]*self.num_layers
        device = x.device
        N, n_ch, n_t = x.shape
        t = torch.LongTensor(c*np.ones(N)).to(device, dtype=torch.int64)
        l = self.eb(t)
        # l.shape: (N, h_ch)
        
        out = self.dropout(x)
        out = md.concat_dim1(out,l)
        out = self.start(out)

        for i, layer in enumerate(self.glu_blocks):
            outl = md.concat_dim1(out,l)
            _out, _state = layer(outl,state=state.pop(0))
            state += [_state]
            out = _out + out
        out = md.concat_dim1(out,l)
        y = self.end(out)

        return y, state



class ConvS2S(nn.Module):
    def __init__(self, enc_src, enc_trg, dec):
        super(ConvS2S, self).__init__()
        self.enc_src = enc_src
        self.enc_trg = enc_trg
        self.dec = dec

    def gaussdis(self, N,mu,sigma):
        nN = np.arange(0,N)
        nN = nN.reshape(N,1)
        x = np.exp(-np.square(nN - mu)/(2.0*sigma**2))
        x = x/np.sum(x,keepdims=True)
        return x

    def localpeak(self, A,mu,sigma):
        epsi = 1e-12
        A = A.flatten()
        N = len(A)
        gw = self.gaussdis(N,mu,sigma).flatten()
        wA = (A+epsi)*gw
        wA = wA/np.sum(wA)
        return np.argmax(wA), wA

    def subsample(self, x, rf):
        device = x.device
        B,D,N = x.shape
        N_mod = int(np.ceil(N/rf))*rf
        if N_mod != N:
            z = torch.tensor(np.zeros((B, D, N_mod-N))).to(device, dtype=torch.float)
            x = torch.cat((x, z), dim=2)
        out = x.permute(0,2,1).reshape(B,N_mod//rf,D*rf).permute(0,2,1)
        return out

    def expand(self, x, rf):
        device = x.device
        B,D,N = x.shape
        out = x.permute(0,2,1).reshape(B,N*rf,D//rf).permute(0,2,1)
        return out

    def pad_zero_frame(self, x):
        device = x.device
        B,D,N = x.shape
        zero = torch.tensor(np.zeros((B,D,1))).to(device, dtype=torch.float)
        out = torch.cat((zero,x),dim=2)
        return out

    def __call__(self, in_s, c_s, in_t, c_t):
        K_s, V_s = self.enc_src(in_s, c_s)

        # K_s.shape: B x d x N
        d = K_s.shape[1]
        Q_t, _ = self.enc_trg(in_t, c_t)
        # Q_t.shape: B x d x T

        # Attention matrix
        # Scaled dot-product attention
        A = F.softmax(torch.matmul(K_s.permute(0,2,1), Q_t)/np.sqrt(d), dim=1)

        # A.shape: B x N x T
        R = torch.matmul(V_s,A)
        # R.shape: B x d x T

        R = torch.cat((R,F.dropout(Q_t, p=0.9)), dim=1)

        y, _ = self.dec(R, c_t)

        return y, A


    def calc_loss(self, x_s, x_t, m_s, m_t, c_s, c_t, pos_weight=1.0,
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

        x_s = self.subsample(x_s,rf)
        x_t = self.subsample(x_t,rf)

        # Pad all-zero frame
        x_t = self.pad_zero_frame(x_t)
        
        B,D,N = x_s.shape
        B,D,T = x_t.shape
        assert D == num_mels*rf
        
        pos_s = md.position_encoding(N, D)
        pos_t = md.position_encoding(T, D)
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
        
        y, A = self(in_s, c_s, in_t, c_t)

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

    def inference(self, x_s, c_s, c_t, rf, pos_weight=1.0, refine='raw'):
        start = time.time()
        
        device = x_s.device
        # x_s.shape: batchsize x num_mels x N
        num_mels = x_s.shape[1]

        x_s = self.subsample(x_s, rf)
        BatchSize,D,N = x_s.shape

        pos_s = md.position_encoding(N, D)
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
            K, V = self.enc_src(in_s,c_s)
        d = K.shape[1]
        
        if refine == 'raw' or refine == None:
            # Raw attention
            T = round(N*2.0)
            in_t = x_t

            pos_t = md.position_encoding(T, D)
            pos_t = torch.tensor(pos_t).to(device, dtype=torch.float)
            pos_t = pos_t.repeat(BatchSize,1,1)

            state_out_enc_trg = None
            state_out_dec = None
            for t in range(0,T):

                in_t[:,0:pos_t.shape[1],:] = in_t[:,0:pos_t.shape[1],:] + pos_t[:,:,t:t+1]/scale_emb * pos_weight
                
                with torch.no_grad():
                    Q, state_out_enc_trg = self.enc_trg(in_t, c_t, state_out_enc_trg)
                    # Scaled dot-product attention
                    A = F.softmax(torch.matmul(K.permute(0,2,1), Q)/np.sqrt(d), dim=1)
                    R = torch.matmul(V,A)
                    R = torch.cat((R,F.dropout(Q, p=0.0, training=False)), dim=1)
                    y, state_out_dec = self.dec(R,c_t, state_out_dec)
                    y_concat = y if t == 0 else torch.cat((y_concat,y), dim=2)
                    A_concat = A if t == 0 else torch.cat((A_concat,A), dim=2)
                    in_t = y

            elapsed_time = time.time() - start
            A_np = A_concat[0,:,:].detach().cpu().clone().numpy()**0.3
            path = self.mydtw_fromDistMat(1.0-A_np,w=100,p=0.1)

            end_of_frame = path[1][-1]
            #end_of_frame = min(path[1][-1]+20, T)
            #end_of_frame = T
                
        elif refine == 'diagonal':
            # Exactly diagonal attention (no time-warping)
            T = N
            end_of_frame = T
            in_t = x_t

            pos_t = md.position_encoding(T, D)
            pos_t = torch.tensor(pos_t).to(device, dtype=torch.float)
            pos_t = pos_t.repeat(BatchSize,1,1)

            state_out_enc_trg = None
            state_out_dec = None
            for t in range(0,T):

                in_t[:,0:pos_t.shape[1],:] = in_t[:,0:pos_t.shape[1],:] + pos_t[:,:,t:t+1]/scale_emb * pos_weight

                with torch.no_grad():
                    Q, state_out_enc_trg = self.enc_trg(in_t, c_t, state_out_enc_trg)
                    R = torch.cat((V[:,:,t:t+1],F.dropout(Q, p=0.0, training=False)), dim=1)
                    y, state_out_dec = self.dec(R,c_t, state_out_dec)
                    y_concat = y if t == 0 else torch.cat((y_concat,y), dim=2)
                    in_t = y

            elapsed_time = time.time() - start
            A_concat = np.eye(N).reshape(1,N,N)
            A_concat = torch.tensor(A_concat).to(device, dtype=torch.float)
            path = [np.arange(N), np.arange(N)]

        elif refine == 'forward':
            # Forward attention
            T = round(N*2.0)
            n_argmax = 0
            y_samples = np.array([0])
            x_samples = np.array([0])
            in_t = x_t

            pos_t = md.position_encoding(T, D)
            pos_t = torch.tensor(pos_t).to(device, dtype=torch.float)
            pos_t = pos_t.repeat(BatchSize,1,1)

            state_out_enc_trg = None
            state_out_dec = None
            for t in range(0,T):

                in_t[:,0:pos_t.shape[1],:] = in_t[:,0:pos_t.shape[1],:] + pos_t[:,:,t:t+1]/scale_emb * pos_weight

                with torch.no_grad():
                    Q, state_out_enc_trg = self.enc_trg(in_t, c_t, state_out_enc_trg)
                    # Scaled dot-product attention
                    A = F.softmax(torch.matmul(K.permute(0,2,1), Q)/np.sqrt(d), dim=1)

                    A_np = A.detach().cpu().clone().numpy()
                    # Prediction of attended time point
                    if t==0:
                        n_argmax_tmp = self.localpeak(A_np,n_argmax,5.0)[0]
                        if n_argmax_tmp > n_argmax:
                            n_argmax = n_argmax_tmp
                    else:
                        n_argmax_tmp = self.localpeak(A_np,n_argmax,5.0)[0]
                        y_samples = np.append(y_samples,n_argmax_tmp)
                        x_samples = np.append(x_samples,t)
                        slope = (np.mean((y_samples-np.mean(y_samples))*(x_samples-np.mean(x_samples)))
                                 /(max(np.std(x_samples),1e-10)**2))
                        n_argmax = int(round(slope*(t+1)))

                    A_np[0,0:max(n_argmax-20//rf,0),0] = 0
                    A_np[0,min(n_argmax+40//rf,N-1):,0] = 0
                    A_np = (np.maximum(A_np,1e-10))/np.sum(np.maximum(A_np,1e-10))
                    A_ = torch.tensor(A_np).to(device, dtype=torch.float)
                    A_concat = A_ if t == 0 else torch.cat((A_concat, A_), dim=2)
                    R = torch.matmul(V,A_)
                    R = torch.cat((R,F.dropout(Q, p=0.0, training=False)), dim=1)
                    y, state_out_dec = self.dec(R,c_t,state_out_dec)
                    y_concat = y if t == 0 else torch.cat((y_concat,y), dim=2)

                    in_t = y

            elapsed_time = time.time() - start
            A_tmp = A_concat[0,:,:].detach().cpu().clone().numpy()**0.3
            #import pdb;pdb.set_trace() # Breakpoint
            path = self.mydtw_fromDistMat(1.0-A_tmp,w=100,p=0.1)
            end_of_frame = path[1][-1]
            #end_of_frame = T
            
        A_out = A_concat[:,:,0:end_of_frame].clone()
        A_out = A_out.detach().cpu().clone().numpy()

        melspec_conv = self.expand(y_concat[:,0:D,0:end_of_frame],rf).detach().cpu().clone().numpy()
        melspec_conv = melspec_conv[0,:,:]

        return melspec_conv, A_out, elapsed_time

    def mydtw_fromDistMat(self, D0, w=np.inf, p=0.0):
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