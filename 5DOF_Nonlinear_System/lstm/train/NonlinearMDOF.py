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
file_tag = 'MDOF_BoundedWhiteNoise_NonlinearVicinity.mat'
interval = 1
x, dx, ddx, t, ddxg = load_data(file_tag, interval, device)
dt = torch.mean(t[1:]-t[:-1])

## set up LSTM
source_seq, target_seq = 32, 16
model = MyLSTM(input_size = x.shape[1]+dx.shape[1]+ddxg.shape[1], hidden_size = 32, output_size = x.shape[1]+dx.shape[1], 
               num_hidd_layers = 3, OutputWindowSize = target_seq).to(device)

## train
start_time = time.time()

epochs = 10000 # 10000
train(model, x, dx, ddxg, epochs, writer, source_seq, target_seq, dt)

## timing
elapsed = time.time() - start_time  
writer.add_text('Time', 'Training time:' + str(elapsed))
 
writer.flush()
writer.close()

#%%