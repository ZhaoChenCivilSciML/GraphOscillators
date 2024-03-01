#%% packages
import scipy.io
import torch 
import os
from utils_NonlinearMDOF import *
from torch.utils.tensorboard import SummaryWriter
import time

## fix random seeds
torch.manual_seed(0)

## log training statistics
# Writer will output to ./runs/ directory by default
writer = SummaryWriter("GNN")

## select device
if torch.cuda.is_available():
    cuda_tag = "cuda:0"
    device = torch.device(cuda_tag)  # ".to(device)" should be added to all models and inputs
    print("Running on " + cuda_tag)
else:
    device = torch.device("cpu")
    print("Running on the CPU")

## load data
file_tag = 'MDOF_ChiChi90_scale20_NonlinearVicinity_14kSteps.mat'
interval = 1
x, dx, ddx, t, ddxg = load_data(file_tag, interval, device)
dt = torch.mean(t[1:]-t[:-1])

## set up LSTM
source_seq, target_seq = 32, 16
model = MyLSTM(input_size = x.shape[1]+dx.shape[1]+ddxg.shape[1], hidden_size = 32, output_size = x.shape[1]+dx.shape[1], 
               num_hidd_layers = 3, OutputWindowSize = target_seq).to(device)

# load trained mdl
checkpoint = torch.load('trained.tar', map_location=device)
model.load_state_dict(checkpoint['model'])

# load normalization metrics
data_tr = scipy.io.loadmat('pred_train.mat')
x_mean = torch.from_numpy(data_tr['x_mean']).float().to(device)
x_std = torch.from_numpy(data_tr['x_std']).float().to(device)
dx_mean = torch.from_numpy(data_tr['dx_mean']).float().to(device)
dx_std = torch.from_numpy(data_tr['dx_std']).float().to(device)
ddxg_mean = torch.from_numpy(data_tr['ddxg_mean']).float().to(device)
ddxg_std = torch.from_numpy(data_tr['ddxg_std']).float().to(device)

## train
start_time = time.time()

epochs = 1 # 10000
train(model, x, dx, ddxg, epochs, writer, source_seq, target_seq, x_mean, x_std, dx_mean, dx_std, ddxg_mean, ddxg_std)

## timing
elapsed = time.time() - start_time  
writer.add_text('Time', 'Training time:' + str(elapsed))
 
writer.flush()
writer.close()

#%%