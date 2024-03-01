
## packages
import torch 
from tqdm import tqdm
import scipy.io
import os

## classes
        
class MyLSTM(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_hidd_layers, OutputWindowSize):
        super(MyLSTM, self).__init__()

        self.input_layer = torch.nn.LSTM(input_size, hidden_size, num_layers = 1 + num_hidd_layers)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)   
        self.OutputWindowSize = OutputWindowSize         
    def forward(self, x):
        h1 = self.input_layer(x)[0] # an LSTM layer outputs a tuple of {output, (hn, cn)}
        
        # only select the last self.OutputWindowSize elements in the sequence for the many-to-some prediction
        # linear layer takes input of dims: (seq_len, batch, input_size) and gives output of dim (seq_len, batch, output_size)
        y = self.output_layer(h1[-self.OutputWindowSize:, :, :]) 
        
        return y


## functions
def train(model, x, dx, ddxg, epochs, writer, source_seq, target_seq, dt):
    optimizer = torch.optim.Adam([{'params': model.parameters()},
                ], lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # normalize data
    x, x_mean, x_std = normalize_data(x)
    dx, dx_mean, dx_std = normalize_data(dx)
    ddxg, ddxg_mean, ddxg_std = normalize_data(ddxg)

    # batch data
    x_t_seq, x_t_1_seq = batch_data(x, source_seq, target_seq)
    dx_t_seq, dx_t_1_seq = batch_data(dx, source_seq, target_seq)
    ddxg_t_seq, ddxg_t_1_seq = batch_data(ddxg, source_seq, target_seq)

    # train & val data split: sequential split
    N_batch = x_t_seq.shape[1]
    split_ind = int(0.7*N_batch)

    x_t_seq_tr, x_t_seq_val, x_t_1_seq_tr, x_t_1_seq_val = split_data(x_t_seq, x_t_1_seq, split_ind)
    dx_t_seq_tr, dx_t_seq_val, dx_t_1_seq_tr, dx_t_1_seq_val = split_data(dx_t_seq, dx_t_1_seq, split_ind)
    ddxg_t_seq_tr, ddxg_t_seq_val, ddxg_t_1_seq_tr, ddxg_t_1_seq_val = split_data(ddxg_t_seq, ddxg_t_1_seq, split_ind)

    X_t_1_seq_tr = torch.cat([x_t_1_seq_tr, dx_t_1_seq_tr], -1)
    X_t_1_seq_val = torch.cat([x_t_1_seq_val, dx_t_1_seq_val], -1)

    for epoch in tqdm(range(epochs)):
        ## training forward pass
        X_t_1_seq_tr_pred = model(torch.cat([x_t_seq_tr, dx_t_seq_tr, ddxg_t_seq_tr], -1))
        loss_X_tr = loss_fn(X_t_1_seq_tr_pred, X_t_1_seq_tr)

        # total loss
        loss_tr = loss_X_tr 

        ## val forward pass
        with torch.no_grad():
            X_t_1_seq_val_pred = model(torch.cat([x_t_seq_val, dx_t_seq_val, ddxg_t_seq_val], -1))
            loss_X_val = loss_fn(X_t_1_seq_val_pred, X_t_1_seq_val)

        writer.add_scalars('loss_X', {'tr':loss_X_tr.item(), 'val':loss_X_val.item()}, epoch)

        optimizer.zero_grad()
    
        loss_tr.backward()
    
        optimizer.step()

    # evaluate model and report metrics
    with torch.no_grad():
        # sequence forecasting
        seq_err_X_tr = torch.norm(X_t_1_seq_tr_pred.flatten() - X_t_1_seq_tr.flatten())/torch.norm(X_t_1_seq_tr.flatten())*100
        seq_err_X_val = torch.norm(X_t_1_seq_val_pred.flatten() - X_t_1_seq_val.flatten())/torch.norm(X_t_1_seq_val.flatten())*100
        writer.add_text('seq_err_X_tr', 'Train Error(%):' + str(seq_err_X_tr.item()))
        writer.add_text('seq_err_X_val', 'Val Error(%):' + str(seq_err_X_val.item()))

        # open-loop forecasting
        X = torch.cat([x, dx], -1)
        X_t_open = X[:source_seq].unsqueeze(1)
        X_t_1_open = X[source_seq:].unsqueeze(1)
        X_t_1_open_pred_list = []

        ddxg_t_open = ddxg[:source_seq].unsqueeze(1) 
        ddxg_t_1_open = ddxg[source_seq:].unsqueeze(1) 

        for nstep in range(X.shape[0]-source_seq-target_seq+1):
            seq_pred = model(torch.cat([X_t_open, ddxg_t_open], -1))
            X_t_1_open_pred_list.append(seq_pred)
            X_t_open = torch.cat([X_t_open[1:], seq_pred[[0]]], 0)
            ddxg_t_open = torch.cat([ddxg_t_open[1:], ddxg_t_1_open[[nstep]]], 0)

        X_t_1_open_pred = torch.cat([seq_pred[[0]] for seq_pred in X_t_1_open_pred_list[:-1]] + [X_t_1_open_pred_list[-1]], 0)
        X_open_err = torch.norm(X_t_1_open_pred.flatten() - X_t_1_open.flatten())/torch.norm(X_t_1_open.flatten())*100
        writer.add_text('X_open_err', str(X_open_err.item()))

    # save data
    scipy.io.savemat('pred.mat',{'X_t_1_seq_tr_pred':X_t_1_seq_tr_pred.detach().cpu().numpy(), 'X_t_1_seq_tr':X_t_1_seq_tr.cpu().numpy(),
    'X_t_1_seq_val_pred':X_t_1_seq_val_pred.detach().cpu().numpy(), 'X_t_1_seq_val':X_t_1_seq_val.cpu().numpy(),
    'X_t_1_open_pred': X_t_1_open_pred.detach().cpu().numpy(), 'X_t_1_open': X_t_1_open.cpu().numpy(),
    'x_mean': x_mean.cpu().numpy(), 'x_std': x_std.cpu().numpy(),

    'dx_mean': dx_mean.cpu().numpy(), 'dx_std': dx_std.cpu().numpy(),

    'ddxg_mean': ddxg_mean.cpu().numpy(), 'ddxg_std': ddxg_std.cpu().numpy(),

    })


    # save model
    save_dict = {}
    save_dict['model'] = model.state_dict()
    torch.save(save_dict, 'trained.tar')

def split_data(X_t_seq, X_t_1_seq, split_ind):
    X_t_seq_tr = X_t_seq[:, :split_ind, :]
    X_t_seq_val = X_t_seq[:, split_ind:, :]

    X_t_1_seq_tr = X_t_1_seq[:, :split_ind, :]
    X_t_1_seq_val = X_t_1_seq[:, split_ind:, :]
    return X_t_seq_tr, X_t_seq_val, X_t_1_seq_tr, X_t_1_seq_val

def batch_data(ah, source_seq, target_seq):
    x_t_seq = torch.stack([ah[i:i+source_seq] for i in range(ah.shape[0]-source_seq-target_seq+1)], dim=1) # (sequence, batch, N_inputs)
    x_t_1_seq = torch.stack([ah[i+source_seq:i+source_seq+target_seq] for i in range(ah.shape[0]-source_seq-target_seq+1)], dim=1)
    return x_t_seq, x_t_1_seq

def normalize_data(ah, x_mean = None, x_std = None):
    if x_mean == None and x_std == None:
        x_mean = torch.mean(ah, dim=0, keepdim=True)
        x_std = torch.std(ah, dim=0, keepdim=True)
    ah = (ah - x_mean)/x_std # (timesteps, state)
    return ah, x_mean, x_std

def de_normalize_data(ah, x_mean, x_std):
    return ah*x_std + x_mean

def load_data(file_tag, interval, device):
    data = scipy.io.loadmat(os.path.join(os.path.dirname(os.getcwd()), file_tag))
    x = torch.from_numpy(data['x'][::interval]).float().to(device) # (timesteps, DOFs)
    dx = torch.from_numpy(data['dx'][::interval]).float().to(device) # (timesteps, DOFs)
    t = torch.from_numpy(data['t'][::interval]).float().to(device) # (timesteps, 1)
    ddxg = torch.from_numpy(data['f'][::interval].T).float().to(device) # (timesteps, 1)
    ddx = torch.from_numpy(data['ddx'][::interval]).float().to(device) # (timesteps, DOFs)
    return x, dx, ddx, t, ddxg

