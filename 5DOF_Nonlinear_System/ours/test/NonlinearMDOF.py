#%%
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import scipy.io
import os
from utils_NonlinearMDOF import *

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
data = scipy.io.loadmat(os.path.join(os.path.dirname(os.getcwd()), 'MDOF_ChiChi90_scale20_NonlinearVicinity_14kSteps.mat'))
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

in_feats, MLP_feats, out_feats, n_layers = source_seq*2, 24, 1, 1 # source_seq*2: x and dx

MLP_layers = 3
gnn = Rel_Mot_GNN(g, in_feats, out_feats, MLP_feats, MLP_layers, n_layers).to(device)

# load trained mdl
checkpoint = torch.load('trained_GNN.tar', map_location=device)
gnn.load_state_dict(checkpoint['gnn_state_dict'])

# load normalization metrics
data_tr = scipy.io.loadmat('pred_train.mat')
dis_mean = torch.from_numpy(data_tr['dis_mean']).double().to(device)
dis_std = torch.from_numpy(data_tr['dis_std']).double().to(device)
vel_mean = torch.from_numpy(data_tr['vel_mean']).double().to(device)
vel_std = torch.from_numpy(data_tr['vel_std']).double().to(device)
acc_mean = torch.from_numpy(data_tr['acc_mean']).double().to(device)
acc_std = torch.from_numpy(data_tr['acc_std']).double().to(device)

# train
epochs = 1 # epochs for each horizon. 10000
Train(gnn, epochs, writer, acc, vel, dis, ag, dt, device, source_seq, target_seq, dis_mean, dis_std, vel_mean, vel_std, acc_mean, acc_std)

# timing
elapsed = time.time() - start_time  
writer.add_text('Time', 'Training time:' + str(elapsed))
 
writer.flush()
writer.close()

#%%