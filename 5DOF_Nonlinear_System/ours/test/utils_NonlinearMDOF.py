import torch
import dgl 
import scipy.io
from tqdm import tqdm

""" functions """
def Train(gnn, epochs, writer, acc, vel, dis, ag, dt, device, source_seq, target_seq, dis_mean = None, dis_std = None, vel_mean = None, vel_std = None,
 acc_mean = None, acc_std = None):
    optimizer = torch.optim.Adam([
                {'params': gnn.parameters()},
            ]
    , lr=1e-3)

    # normalize graph inputs and outputs
    if dis_mean == None:
        dis_mean = torch.mean(dis, 1, True).unsqueeze(0) # (1, states, 1) for graph computation
        dis_std = torch.std(dis, 1, keepdim=True).unsqueeze(0)

        vel_mean = torch.mean(vel, 1, True).unsqueeze(0) 
        vel_std = torch.std(vel, 1, keepdim=True).unsqueeze(0)

        acc_mean = torch.mean(acc, 1, True).unsqueeze(0) 
        acc_std = torch.std(acc, 1, keepdim=True).unsqueeze(0)

    # batch data. (state, timesteps) -> (batches, state, seq). moving horizon
    dis_t_seq, dis_t_1_seq = batch_data(dis, source_seq, target_seq) 
    vel_t_seq, vel_t_1_seq = batch_data(vel, source_seq, target_seq) 
    ag_t_seq, ag_t_1_seq = batch_data(ag, source_seq, target_seq) 

    # train & val data split
    N_batches = dis_t_seq.shape[0] # mini-batch training does not boost the speed

    # split_ind = torch.randperm(N_batches).to(device)
    split_ind = torch.arange(N_batches).to(device)
    tr_ind = split_ind[:int(0.7*N_batches)]
    val_ind = split_ind[int(0.7*N_batches):]

    dis_t_seq_tr, dis_t_seq_val, dis_t_1_seq_tr, dis_t_1_seq_val = data_split(dis_t_seq, dis_t_1_seq, tr_ind, val_ind)

    vel_t_seq_tr, vel_t_seq_val, vel_t_1_seq_tr, vel_t_1_seq_val = data_split(vel_t_seq, vel_t_1_seq, tr_ind, val_ind)

    ag_t_seq_tr, ag_t_seq_val, ag_t_1_seq_tr, ag_t_1_seq_val = data_split(ag_t_seq, ag_t_1_seq, tr_ind, val_ind)

    loss_fn = torch.nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        # train forward
        dis_t_1_seq_tr_pred, vel_t_1_seq_tr_pred = GNN_forecast(dis_t_seq_tr, vel_t_seq_tr, ag_t_seq_tr[..., [-1]], ag_t_1_seq_tr, dt, target_seq, gnn, dis_mean, dis_std,
         vel_mean, vel_std, acc_mean, acc_std) # (batches, state, target seq)

        # data loss
        dis_loss = loss_fn(normalize(dis_t_1_seq_tr_pred, dis_mean, dis_std), normalize(dis_t_1_seq_tr, dis_mean, dis_std)) 
        vel_loss = loss_fn(normalize(vel_t_1_seq_tr_pred, vel_mean, vel_std), normalize(vel_t_1_seq_tr, vel_mean, vel_std)) 
         
        tr_loss = dis_loss + vel_loss

        # backpropagation
        # optimizer.zero_grad()
        # tr_loss.backward()
        # optimizer.step()

        # val forward
        with torch.no_grad():
            dis_t_1_seq_val_pred, vel_t_1_seq_val_pred = GNN_forecast(dis_t_seq_val, vel_t_seq_val, ag_t_seq_val[..., [-1]], ag_t_1_seq_val, dt, target_seq, gnn,
             dis_mean, dis_std,
                vel_mean, vel_std, acc_mean, acc_std) # (batches, state, target seq)

            # data loss
            dis_loss_val = loss_fn(normalize(dis_t_1_seq_val_pred, dis_mean, dis_std), normalize(dis_t_1_seq_val, dis_mean, dis_std)) 
            vel_loss_val = loss_fn(normalize(vel_t_1_seq_val_pred, vel_mean, vel_std), normalize(vel_t_1_seq_val, vel_mean, vel_std)) 

        # log metrics
        writer.add_scalars('dis_loss', {'tr':dis_loss.item(), 'val': dis_loss_val.item()}, epoch)
        writer.add_scalars('vel_loss', {'tr':vel_loss.item(), 'val': vel_loss_val.item()}, epoch)

    # evaluate all batches and report final metrics
    dis_err_tr = torch.norm(dis_t_1_seq_tr_pred.flatten() - dis_t_1_seq_tr.flatten())/torch.norm(dis_t_1_seq_tr.flatten())*100
    vel_err_tr = torch.norm(vel_t_1_seq_tr_pred.flatten() - vel_t_1_seq_tr.flatten())/torch.norm(vel_t_1_seq_tr.flatten())*100

    dis_err_val = torch.norm(dis_t_1_seq_val_pred.flatten() - dis_t_1_seq_val.flatten())/torch.norm(dis_t_1_seq_val.flatten())*100
    vel_err_val = torch.norm(vel_t_1_seq_val_pred.flatten() - vel_t_1_seq_val.flatten())/torch.norm(vel_t_1_seq_val.flatten())*100

    writer.add_text('dis_err_tr', 'Train Error(%):' + str(dis_err_tr.item()))
    writer.add_text('vel_err_tr', 'Train Error(%):' + str(vel_err_tr.item()))

    writer.add_text('dis_err_val', 'Val Error(%):' + str(dis_err_val.item()))
    writer.add_text('vel_err_val', 'Val Error(%):' + str(vel_err_val.item()))

    # open-loop
    with torch.no_grad():
        dis_open_pred, vel_open_pred = GNN_forecast(dis_t_seq_tr[[0]], vel_t_seq_tr[[0]], (ag_t_seq_tr[[0]])[...,[-1]], ag[...,source_seq:].unsqueeze(0), dt,
         dis.shape[1] - source_seq, gnn, dis_mean, dis_std, vel_mean, vel_std, acc_mean, acc_std) # (1, state, total timesteps - source seq)

    dis_err_open = torch.norm(dis_open_pred.flatten() - dis[...,source_seq:].flatten())/torch.norm(dis[...,source_seq:].flatten())*100
    vel_err_open = torch.norm(vel_open_pred.flatten() - vel[...,source_seq:].flatten())/torch.norm(vel[...,source_seq:].flatten())*100
    writer.add_text('dis_err_open', 'Open Error(%):' + str(dis_err_open.item()))
    writer.add_text('vel_err_open', 'Open Error(%):' + str(vel_err_open.item()))

    # save data
    scipy.io.savemat('pred.mat',{
    'dis_tr_pred':dis_t_1_seq_tr_pred.detach().cpu().numpy(), 'dis_tr':dis_t_1_seq_tr.cpu().numpy(),
    'vel_tr_pred':vel_t_1_seq_tr_pred.detach().cpu().numpy(), 'vel_tr':vel_t_1_seq_tr.cpu().numpy(),

    'dis_val_pred':dis_t_1_seq_val_pred.detach().cpu().numpy(), 'dis_val':dis_t_1_seq_val.cpu().numpy(),
    'vel_val_pred':vel_t_1_seq_val_pred.detach().cpu().numpy(), 'vel_val':vel_t_1_seq_val.cpu().numpy(),

    'dis_open_pred':dis_open_pred.detach().cpu().numpy(), 'dis_open':dis[...,source_seq:].cpu().numpy(),
    'vel_open_pred':vel_open_pred.detach().cpu().numpy(), 'vel_open':vel[...,source_seq:].cpu().numpy(),

    'dis_mean': dis_mean.cpu().numpy(), 'dis_std': dis_std.cpu().numpy(),
    'vel_mean': vel_mean.cpu().numpy(), 'vel_std': vel_std.cpu().numpy(),
    'acc_mean': acc_mean.cpu().numpy(), 'acc_std': acc_std.cpu().numpy(),
    })

    # save graph
    # save_dict = {}
    # save_dict['gnn_state_dict'] = gnn.state_dict()
    # torch.save(save_dict, 'trained_GNN.tar')

def data_split(dis_t_seq, dis_t_1_seq, tr_ind, val_ind):
    dis_t_seq_tr = dis_t_seq[tr_ind]
    dis_t_seq_val = dis_t_seq[val_ind]
    dis_t_1_seq_tr = dis_t_1_seq[tr_ind]
    dis_t_1_seq_val = dis_t_1_seq[val_ind]
    return dis_t_seq_tr, dis_t_seq_val, dis_t_1_seq_tr, dis_t_1_seq_val

def batch_data(ah, source_seq, target_seq):
    # ah: (states, timesteps)
    x_t_seq = torch.stack([ah[..., i:i+source_seq] for i in range(ah.shape[-1]-source_seq-target_seq+1)], dim=0) # (batch, states, sequence)
    x_t_1_seq = torch.stack([ah[..., i+source_seq:i+source_seq+target_seq] for i in range(ah.shape[-1]-source_seq-target_seq+1)], dim=0)
    return x_t_seq, x_t_1_seq

def normalize(x, x_mean, x_std):
    return (x - x_mean)/x_std 

def de_normalize(x_norm, x_mean, x_std):
    return x_norm*x_std + x_mean 

def GNN_forecast(dis_t_seq, vel_t_seq, ag_t, ag_t_1_seq, dt, target_seq, gnn, dis_mean, dis_std, vel_mean, vel_std, acc_mean, acc_std):
    # dis_t_seq: (batches, states, source seq)
    # vel_t: (batches, states, 1)
    # ag_t: (batches, states, 1)
    dis_t_1_seq_pred_list = []
    vel_t_1_seq_pred_list = []
    ag_seq = torch.cat([ag_t, ag_t_1_seq], -1) # (batches, states, target_seq+1)
    for nstep in range(target_seq):
        ag_t = ag_seq[..., [nstep]]
        ag_t_1 = ag_seq[..., [nstep+1]]
        dis_t_1, vel_t_1 = timestepper_forward(dis_t_seq, vel_t_seq, ag_t, ag_t_1, dt, gnn, dis_mean, dis_std, vel_mean, vel_std, acc_mean, acc_std)
        dis_t_1_seq_pred_list.append(dis_t_1) # next step prediction. (batches, states, 1)
        vel_t_1_seq_pred_list.append(vel_t_1) # next step prediction. (batches, states, 1)

        # skip the first timestep and concatenate with a new one
        dis_t_seq = torch.cat([dis_t_seq[..., 1:], dis_t_1], dim=-1)
        vel_t_seq = torch.cat([vel_t_seq[..., 1:], vel_t_1], dim=-1)


    dis_t_1_seq_pred = torch.cat(dis_t_1_seq_pred_list, dim=-1)
    vel_t_1_seq_pred = torch.cat(vel_t_1_seq_pred_list, dim=-1)
    return dis_t_1_seq_pred, vel_t_1_seq_pred # (batches, states, target seq)

def timestepper_forward(dis_t_seq, vel_t_seq, ag_t, ag_t_1, dt, gnn, dis_mean, dis_std, vel_mean, vel_std, acc_mean, acc_std):
    # dis_t_seq: (batches, states, seq)
    # vel_t: (batches, states, 1)
    # ag_t: (batches, states, 1)

    # predictor: forward euler
    dis_t = dis_t_seq[...,[-1]]
    vel_t = vel_t_seq[...,[-1]]
    dis_t_1 = dis_t + dt*vel_t

    acc_t = Pi_graph(dis_t_seq, vel_t_seq, ag_t, gnn, dis_mean, dis_std, vel_mean, vel_std, acc_mean, acc_std)

    vel_t_1 = vel_t + dt*acc_t

    # corrector: implicit midpoint method
    dis_t_1 = dis_t + 0.5*dt*(vel_t + vel_t_1) 

    dis_t_1_seq = torch.cat([dis_t_seq[..., 1:], dis_t_1], -1)
    vel_t_1_seq = torch.cat([vel_t_seq[..., 1:], vel_t_1], -1)

    acc_t_1 = Pi_graph(dis_t_1_seq, vel_t_1_seq, ag_t_1, gnn, dis_mean, dis_std, vel_mean, vel_std, acc_mean, acc_std)
    vel_t_1 = vel_t + 0.5*dt*(acc_t + acc_t_1)

    return dis_t_1, vel_t_1

def Pi_graph(dis_t_seq, vel_t_seq, ag_t, gnn, dis_mean, dis_std, vel_mean, vel_std, acc_mean, acc_std):
    # dis_t_seq: (batches, states, seq)
    # dis_mean: (states, 1)
    # ag_t: (batches, states, 1)
    dis_t_seq_norm = normalize(dis_t_seq, dis_mean, dis_std) # (batches, states, seq)
    vel_t_seq_norm = normalize(vel_t_seq, vel_mean, vel_std) # (batches, states, seq)
    gnn_input = torch.cat([dis_t_seq_norm, vel_t_seq_norm], -1) # (batches, states, seq*2)
    acc_GNN = de_normalize(gnn(gnn_input), acc_mean, acc_std) # (batches, states, seq*2) -> (batches, states, hidd_features) -> (batches, states, 1)
    acc_t = acc_GNN + ag_t
    return acc_t # (batches, states, 1)

""" classes """
class Rel_Mot_GNN(torch.nn.Module):
    def __init__(self, g, in_feats, out_feats, MLP_feats, MLP_layers, n_layers):
        super().__init__()
        self.layers_list = torch.nn.ModuleList([Rel_Mot_layer(g, in_feats, out_feats, MLP_feats, MLP_layers) for _ in range(n_layers)])
        
    def forward(self, h):
        # h: (batches, N_V, h_feats)
        h = h.permute((1, 0, 2))
        for layer in self.layers_list:
            h = layer(h) # (N_V, batches, h_feats)
        return h.permute((1, 0, 2)) # (batches, N_V, h_feats)

class Rel_Mot_layer(torch.nn.Module):
    def __init__(self, g, in_feats, out_feats, MLP_feats, MLP_layers):
        super().__init__()
        self.g = g
        no_edge_pairs = g.number_of_edges()//2
        self.MLP_interactive_list = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((no_edge_pairs, in_feats*2, MLP_feats)))] + \
            [torch.nn.Parameter(torch.zeros((no_edge_pairs, MLP_feats, MLP_feats))) for _ in range(MLP_layers)] + \
                [torch.nn.Parameter(torch.zeros((no_edge_pairs, MLP_feats, out_feats)))])

        self.MLP_interactive_bias_list = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((no_edge_pairs, 1, MLP_feats)))] + \
            [torch.nn.Parameter(torch.zeros((no_edge_pairs, 1, MLP_feats))) for _ in range(MLP_layers)] + \
                [torch.nn.Parameter(torch.zeros((no_edge_pairs, 1, out_feats)))])

        for para in self.MLP_interactive_list:
            torch.nn.init.xavier_normal_(para)

        self.MLP_fixed_list = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((1, in_feats*2, MLP_feats)))] + \
            [torch.nn.Parameter(torch.zeros((1, MLP_feats, MLP_feats))) for _ in range(MLP_layers)] + \
                [torch.nn.Parameter(torch.zeros((1, MLP_feats, out_feats)))])

        self.MLP_fixed_bias_list = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((1, 1, MLP_feats)))] + \
            [torch.nn.Parameter(torch.zeros((1, 1, MLP_feats))) for _ in range(MLP_layers)] + \
                [torch.nn.Parameter(torch.zeros((1, 1, out_feats)))])

        for para in self.MLP_fixed_list:
            torch.nn.init.xavier_normal_(para)

        self.swish = torch.nn.SiLU()

    def edge_applying(self, edges):
        # edges.dst['h'].shape: (N_E, batches, h_feats)
        e = torch.cat([edges.dst['h'], edges.src['h']], -1)

        effective_e = e[:e.shape[0]//2] # only process edge pairs
        fixed_e = e[[-1]] # fixed node

        # multiply with edgewise weights for each edge pair
        for counter, weight in enumerate(self.MLP_interactive_list[:-1]):
            effective_e = self.swish(effective_e@weight + self.MLP_interactive_bias_list[counter])  # (N_E//2, batches, h_feats)
        effective_e = effective_e@self.MLP_interactive_list[-1] + self.MLP_interactive_bias_list[-1]

        # mlp for fixed node
        for counter, weight in enumerate(self.MLP_fixed_list[:-1]):
            fixed_e = self.swish(fixed_e@weight + self.MLP_fixed_bias_list[counter])  # (1, batches, h_feats)
        fixed_e = fixed_e@self.MLP_fixed_list[-1] + self.MLP_fixed_bias_list[-1]

        e = torch.cat([effective_e, -effective_e, fixed_e], 0) # enforce anti-symmetricity
        return {'e': e}

    def forward(self, h):
        with self.g.local_scope():
            self.g.ndata['h'] = h
            self.g.apply_edges(self.edge_applying)

            self.g.update_all(dgl.function.copy_e('e', '_'), dgl.function.sum('_', 'z')) # self.g.ndata['z']: (N_V, batches, h_feats))
            return self.g.ndata['z']  # (N_V, batches, h_feats))


