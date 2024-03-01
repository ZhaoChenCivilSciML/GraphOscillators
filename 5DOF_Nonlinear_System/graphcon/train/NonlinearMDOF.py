#%%
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import scipy.io
import os
from utils_NonlinearMDOF_dgl import *

if torch.cuda.is_available():
    cuda_tag = "cuda:0"
    device = torch.device(cuda_tag)  # ".to(device)" should be added to all models and inputs
    print("Running on " + cuda_tag)
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
# Writer will output to ./runs/ directory by default
writer = SummaryWriter("GNN")

start_time = time.time()

# fix random seed
np.random.seed(0)
torch.manual_seed(0)

# set default dtype to double to guarantee matrix multiplication accuracy.
torch.set_default_dtype(torch.float64) 

# load data
data = scipy.io.loadmat(os.path.join(os.path.dirname(os.getcwd()), 'MDOF_BoundedWhiteNoise_NonlinearVicinity.mat'))
acc = torch.from_numpy(data['ddx']).double().to(device).t() # (DOFs, timesteps)
vel = torch.from_numpy(data['dx']).double().to(device).t() # (DOFs, timesteps)
dis = torch.from_numpy(data['x']).double().to(device).t() # (DOFs, timesteps)
ag = torch.from_numpy(data['f']).double().to(device) # (1, timesteps)
t = torch.from_numpy(data['t']).double().to(device).t() # (1, timesteps)
NDOF = acc.shape[0]
dt = torch.mean(t[:, 1:] - t[:, :-1], 1).item()

# format graph
g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
g = dgl.add_reverse_edges(g).to(device) # undirected graph
g.add_edges([0], [0]) # add self loop for the fixed node


#%%
# set up the mdls
source_seq = 1
target_seq = 16

in_feats, MLP_feats, out_feats = source_seq*2, 24, 1 # source_seq*2: x and dx

# https://github.com/tk-rusch/GraphCON/blob/main/src/Superpixels/models.py
num_heads = 4
n_MP  = 5 # number of message passings
dropout_rate = 0.05
gnn = My_GraphCON_GAT(in_feats, MLP_feats, out_feats, num_heads, n_MP, g, dropout_rate).to(device)

# train
epochs = 10000 # epochs for each horizon. 20000
Train(gnn, epochs, writer, acc, vel, dis, ag, dt, device, source_seq, target_seq)

# timing
elapsed = time.time() - start_time  
writer.add_text('Time', 'Training time:' + str(elapsed))
 
writer.flush()
writer.close()

#%%
