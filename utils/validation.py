# Copyright (c) 2019, Firmenich SA (Fabio Capela)
# Copyright (c) 2019, Firmenich SA (Guillaume Godin)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import division
from __future__ import unicode_literals
# usual imports
import numpy as np
import pandas as pd
import torch
import statistics
import random
import tqdm
# rdkit imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdEHTTools
# pytorch imports
from torch.nn import MSELoss
from torch.utils.data.sampler import SubsetRandomSampler
# pytorch geometric imports
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
# sklearn imports
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# loads the function for the features
from .features import *


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss


def prepare_data(f_train, target_vec, num_atoms=True, use_hydrogen_bonding=False, use_acid_base = False):
    """
    This function prepares the dataset to be loaded into Pytorch dataloader
     - creates the features per molecule
     - different optional features are added
         1. num_atoms: adds the number of atoms per molecule. It has been used in the paper
         2. use_hydrogen_bonding: adds the hydrogen bonding molecules 
         3. use_acid_base: adds the acid_base molecules
    """
    
    m_train = []
    target_train = dict()
    for x in range(0, len(target_vec)):
        target_train[x] = []
    
    for i in range(len(f_train)):
        m_train.append(Chem.MolFromSmiles(f_train['smiles'][i]))
        for x in range(0, len(target_vec)):
            target_train[x].append(f_train[target_vec[x]][i])

    
    train_X = [mol2vec(x, num_atoms=num_atoms, use_hydrogen_bonding=use_hydrogen_bonding,
                       use_acid_base = use_acid_base) for x in m_train]
    
    for i, data in enumerate(train_X):
        for x in range(0, len(target_vec)):
            data['y%s' %x] = torch.tensor([target_train[x][i]], dtype=torch.float)
                
    return train_X


def isnan(x):
    """ Simple utility to see what is NaN """
    return x!=x

        
def myloss(output_vec, target_vec):
    """ Main Loss that is used for MulitTargets"""
    criterion = torch.nn.MSELoss()
    mse_part = 0
    masks = dict()
    loss1 = dict()
    for x in range(0,len(target_vec)):
        masks[x] = isnan(target_vec[x]) 
        if target_vec[x][~masks[x]].nelement() == 0:
            loss1[x] = torch.sqrt(torch.tensor(1e-20))
            continue
        else: # non nans
            mse_part += criterion(output_vec[x][~masks[x]],target_vec[x][~masks[x]])
            loss1[x] = torch.sqrt(criterion(output_vec[x][~masks[x]],target_vec[x][~masks[x]])+1e-16)
    
    loss = torch.sqrt(mse_part)
    mylist = [loss]
    for x in range(0, len(target_vec)):
        mylist.append(loss1[x]) 
    return mylist


def clean_print(run, string_vec, loss_vec):
    """ Printing function that is used at the end of each CV """
    str1 = "Run %i : Total RMSE: %0.3f" %(run, loss_vec[0])
    for x in range(len(string_vec)):
        str1 += " | %s Loss %0.3f" %(string_vec[x], loss_vec[x+1])
    print(str1)

        
def overall_clean_print(string_vec, loss_vec, std_vec):
    """ Printing function that prints the aggregated results """
    str1 = "Overall RMSE on test: %0.3f +/- %0.2f" %(loss_vec[0], std_vec[0])
    for x in range(len(string_vec)):
        str1 += " | RMSE on %s: %0.3f +/- %0.2f " %(string_vec[x], loss_vec[x+1], std_vec[x+1])
    print(str1)

    
def CrossValidation(mymodel, f_train, test_loader,  n_splits, n_epochs, target_vec, string_vec, batchs, 
                    patience_early=40, patience=10, lr_decay=0.5, num_atoms=True, use_hydrogen_bonding=False, 
                    use_acid_base=False, name_model='mymodel'): 

    """
    Function that does the cross-validation: trains the model CV times, applies the trained model to both 
    the validation set and test set. The test has been previously splitted. Therefore, no internal splitting 
    is performed. 
    """


    cv = KFold(n_splits=n_splits)
    run = 1
    test_loss_vec = []
    loss_vec = dict()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    
    X_train_all = prepare_data(f_train, target_vec, num_atoms=num_atoms, use_hydrogen_bonding=use_hydrogen_bonding, 
                 use_acid_base=use_acid_base)
    
    for x in range(0, len(target_vec)):
        loss_vec[target_vec[x]] = []
        
    for train_index, val_index in cv.split(X_train_all):

        # call the model and send it to the device
        model = mymodel
        model = model.to(device)
        
        # define optimizer, scheduler, early_stopping
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay,
                                                               verbose=False)
        early_stopping = EarlyStopping(patience=patience_early, verbose=False)

        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(val_index)
        train_loader = DataLoader(X_train_all, batch_size=batchs,  sampler=train_sampler)
        val_loader = DataLoader(X_train_all, batch_size=batchs,  sampler=valid_sampler)
        

        for epoch in range(1, n_epochs):
            train_loss = train(train_loader, optimizer, model, target_vec)
            val_loss   = val(val_loader, model, target_vec)
            print('epoch %i: normalized train loss %0.2f val loss %0.2f' %(epoch, train_loss, val_loss), end="\r")
            scheduler.step(train_loss)
            early_stopping(val_loss)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # saving the model at the end of each CV
        torch.save(mymodel.state_dict(), 'model_checkpoint/%s.pk' %name_model)

        # evaluate the model on the test set
        test_loss_list = test(test_loader, model, target_vec)
        test_loss_vec.append(test_loss_list[0])
        for x in range(len(target_vec)):
            loss_vec[target_vec[x]].append(test_loss_list[x+1].item())
        
        clean_print(run, string_vec, test_loss_list)
        run += 1

    # compute the overall results
    mean_loss = np.mean(test_loss_vec)
    std_loss = np.std(test_loss_vec)

    # print the overall results
    str1 = "Overall RMSE on test: %0.3f +/- %0.2f" %(mean_loss, std_loss)
    for x in range(len(target_vec)):
        str1 += " | RMSE on %s: %0.3f +/- %0.2f " %(string_vec[x], np.mean(loss_vec[target_vec[x]]), 
                                                   np.std(loss_vec[target_vec[x]]))
    
    print(" ")
    print('-'*20+'Model Cross-Validation'+'-'*20)
    print(str1)

        
def nonmissingvales(loader, target_num):
    """ function that computes the amount of molecules that do have a specific target """
    count = 0
    for data in loader:
        count +=isnan(data['y%s'%target_num]).sum()
    return len(loader.dataset) - count


def train(loader, optimizer, model, target_vec):
    """ Main function to train the model """
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    loss_all = 0
    output_vec = []
    tar_vec = []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = myloss([output[:,x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])[0]
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(loader.dataset)


def test(loader, model, target_vec):
    """ Main function to test the model """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    loss_all = 0
    loss1_all = dict()
    output_vec1 = []
    tar_vec1 = []
    for x in range(len(target_vec)):
        loss1_all[x]= 0
        
    for data in loader:
        data = data.to(device)
        output = model(data)
        masks0 = isnan(data['y0']) 
        output_vec1.append(output[:,0][~masks0])
        tar_vec1.append(data['y0'][~masks0])
        losslist = myloss([output[:,x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])
        loss_all += losslist[0].item() * data.num_graphs
        for x in range(len(target_vec)):
            loss1_all[x] += losslist[x+1] * (data.num_graphs - isnan(data['y%s'%x]).sum().item())
    
    
    mylist = [loss_all / len(loader.dataset)]
    for x in range(len(target_vec)):
        mylist.append(loss1_all[x] /nonmissingvales(loader, x).float())
    return mylist


def val(loader, model, target_vec):
    """ Main function to validate the model """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    loss_all = 0
    loss1_all = dict()
    output_vec1 = []
    tar_vec1 = []
    for x in range(len(target_vec)):
        loss1_all[x]= 0
    
    c = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        loss = myloss([output[:,x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])[0]
        loss_all += loss.item() * data.num_graphs
        c += data.y0.size(0)
    return loss_all/c
                
    

    
