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


# Import functions used everywhere
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Sequential as Seq, Linear
from torch_geometric.nn import GINConv, GATConv, global_mean_pool
from torch_geometric.data import DataLoader, Batch


class GGRNet(torch.nn.Module):
    def __init__(self, n_features, n_outputs, dim=95, n_iterations=10):
        super(GGRNet, self).__init__()
        # Preparation of the Graph Isomorphism Convolutional Layer
        self.n_iterations = n_iterations
        nn1 = Seq(Linear(n_features, n_features), ReLU(), Linear(n_features, n_features))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(n_features)
        # Preparation of the Fully Connected Layer 
        self.lin1 = Linear(2*n_features, int(n_features/2.))
        self.lin2 = Linear(2*n_features, int(n_features/2.))
        # Preparation of the MLP
        self.MLP = Seq(Linear(n_features, dim), ReLU(), Linear(dim, n_outputs))
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Graph Isomorphism Convolutional Layer
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.bn1(x1)
        x_new = global_mean_pool(x1, batch)
        h = x_new

        # Recurrence with the two Fully Connected Layers
        for ii in range(self.n_iterations):
            p = self.lin1(torch.cat([x_new, h], dim=1))
            q = self.lin2(torch.cat([x_new, h], dim=1))
            h = torch.cat([F.tanh(q), F.sigmoid(p)], dim=1)

        # Output with the MLP
        output = self.MLP(h)   
        return output
