import numpy as np
import os
import logging
import torch
import itertools

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

    dr = model_config['dropout_ratio']
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

    print("===================================Start Training===================================")
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

            logging.info(model_dir)
            if iml:
                mainloss_mean = 0
                daloss_mean = 0
                for s in range(n_spk):
                    MainLoss, DALoss, A = models['convs2s'].calc_loss(xin[s], xin[s], mask[s], mask[s], s, s, dr, pw, gw, rf)
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

            # List of speaker pairs
            spk_pair_list = perm(n_spk)
            n_spk_pair = len(spk_pair_list)

            mainloss_mean = 0
            daloss_mean = 0
            # Iterate through all speaker pairs
            for m in range(n_spk_pair):
                s0 = spk_pair_list[m][0]
                s1 = spk_pair_list[m][1]
                MainLoss, DALoss, A = models['convs2s'].calc_loss(xin[s0], xin[s1], mask[s0], mask[s1], s0, s1, dr, pw, gw, rf)
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

            b += 1

        if epoch % snapshot == 0:
            for tag in ['enc_src', 'enc_trg', 'dec']:
                #print('save {} at {} epoch'.format(tag, epoch))
                torch.save({'epoch': epoch,
                            'model_state_dict': models[tag].state_dict(),
                            'optimizer_state_dict': optimizers[tag].state_dict()},
                            os.path.join(model_dir, '{}.{}.pt'.format(epoch, tag)))

    print("===================================Training Finished===================================")