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
data = scipy.io.loadmat(os.path.join(os.path.dirname(os.getcwd()), 'MDOF_BoundedWhiteNoise_NonlinearVicinity.mat'))
acc = torch.from_numpy(data['ddx']).double().to(device).t() # (DOFs, timesteps)
vel = torch.from_numpy(data['dx']).double().to(device).t() # (DOFs, timesteps)
dis = torch.from_numpy(data['x']).double().to(device).t() # (DOFs, timesteps)
ag = torch.from_numpy(data['f']).double().to(device) # (1, timesteps)
t = torch.from_numpy(data['t']).double().to(device).t() # (1, timesteps)
NDOF = acc.shape[0]
dt = torch.mean(t[:, 1:] - t[:, :-1], 1).item()


#%%
# set up the mdls
source_seq = 1
target_seq = 16

in_feats, MLP_feats, out_feats = source_seq*2*NDOF, 24, NDOF # source_seq*2: x and dx


gnn = torch.nn.Sequential(torch.nn.Linear(in_feats, MLP_feats),
                          torch.nn.SiLU(),
                          torch.nn.Linear(MLP_feats, MLP_feats),
                          torch.nn.SiLU(),
                          torch.nn.Linear(MLP_feats, MLP_feats),
                          torch.nn.SiLU(),
                          torch.nn.Linear(MLP_feats, MLP_feats),
                          torch.nn.SiLU(),
                          torch.nn.Linear(MLP_feats, out_feats)).to(device)

# train
epochs = 10000 # epochs for each horizon. 10000
Train(gnn, epochs, writer, acc, vel, dis, ag, dt, device, source_seq, target_seq)

# timing
elapsed = time.time() - start_time  
writer.add_text('Time', 'Training time:' + str(elapsed))
 
writer.flush()
writer.close()

#%%