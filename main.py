
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



import os
import sys
import glob
from utils import *
from models import *
import pandas as pd
import configparser
import random
from distutils import util


def prepare_dataset(tar_list):
    cwd = os.getcwd()
    df_blank = pd.DataFrame({'smiles':[]})
    for dataset in tar_list:
        df0 = pd.read_csv(cwd+'/'+'data/'+dataset)
        df_blank =  pd.merge(df_blank, df0, on='smiles', how='outer')
    return df_blank



if __name__ == '__main__':
    
    config = configparser.ConfigParser()
    config.read('config.cfg')
    seed = int(config['parameters']['seed'])
    cwd = os.getcwd()

    seed = int(config['parameters']['seed'])
    nbr_task = int(config['parameters']['number_tasks'])
    dlist = config['parameters']['task_names'].split(',',nbr_task)
    dfnew = prepare_dataset(dlist)
    
    # split train/test 8:2 with random seed
    f_train = dfnew.sample(frac=0.8, random_state=seed)
    f_test  = dfnew[~dfnew.index.isin(f_train.index)]
    f_train = f_train.reset_index()
    f_test  = f_test.reset_index()

    # create the target list (column name)
    target_list = []
    for tar in dlist:
        target_list.append(tar.split('.',1)[0])


    # get the parameters from the config file
    num_atoms            = util.strtobool(config['parameters']['num_atoms'])
    use_hydrogen_bonding = util.strtobool(config['parameters']['use_hydrogen_bonding'])
    use_acid_base       = util.strtobool(config['parameters']['use_acid_base'])
    n_splits             = int(config['parameters']['n_splits'])
    dim                  = int(config['parameters']['dim'])
    n_epochs             = int(config['parameters']['n_epochs'])
    n_batchs             = int(config['parameters']['batch'])
    n_iterations         = int(config['parameters']['n_iterations'])
    patience             = int(config['parameters']['patience'])
    patience_early       = int(config['parameters']['patience_early'])
    lr_decay             = float(config['parameters']['lr_decay'])

    
    # prepare data to create test_loader
    test_X = prepare_data(f_test, target_list,
                          num_atoms = num_atoms,
                          use_hydrogen_bonding=use_hydrogen_bonding,
                          use_acid_base=use_acid_base)
    
    test_loader = DataLoader(test_X, batch_size=len(f_test), shuffle=False, drop_last=False)

    # compute the number of features available
    for data in test_loader:
        n_features = data.x.size(1)
        break

    print('-'*60)
    print("Number of features to be used: %i" %n_features)
    print('-'*60)
    
    # select the correct model
    modelstr = config['parameters']['model']
    if modelstr == 'GIN':
        model0 = GINet(n_features, n_outputs=len(target_list), dim=dim)

    elif modelstr == 'GGRNet':
        model0 = GGRNet(n_features, n_outputs=len(target_list), dim=dim, n_iterations=n_iterations)
        
    elif modelstr == 'GAIN':
        model0 = GINAttNet(n_features, n_outputs=len(target_list), dim=dim)
    else:
        print('You did not put a correct model')
        print('Available models: GIN, GGRNet, GAIN')
        print('')
        print('Try Again')
        sys.exit()


    # cross-validation
    CrossValidation(model0, f_train, test_loader,  n_splits, n_epochs, target_list, target_list, n_batchs, 
                    patience_early=patience_early, patience=patience, lr_decay=lr_decay,
                    num_atoms = num_atoms, use_hydrogen_bonding=use_hydrogen_bonding,
                    use_acid_base=use_acid_base, name_model='mymodel')
