import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMB_SIZE = 12
HIDDEN_SIZE = 96
BATCH_SIZE = 16

class MyDataset(Dataset):

    def __init__(self, csv_file):
        self.csv = np.array(pd.read_csv(csv_file, sep=',', header=None))
        self.csv = torch.Tensor([[[get_start_embeds(int(x)) for x in my_input] for my_input in problem] for problem in self.csv]).long()
    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        # res = torch.Tensor([[int(x) for x in sudoku] for sudoku in self.csv[idx]]).long()
        # return res
        return self.csv[idx]

def get_edges():
    def cross(a):
        return [(i, j) for i in a.flatten() for j in a.flatten() if not i == j]

    idx = np.arange(81).reshape(9, 9)
    rows, columns, squares = [], [], []
    for i in range(9):
        rows += cross(idx[i, :])
        columns += cross(idx[:, i])
    for i in range(3):
        for j in range(3):
            squares += cross(idx[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3])
    return torch.Tensor(list(set(rows + columns + squares))).long()

def get_start_embeds(embed, start_state):
    start_state = torch.Tensor([x.item() for x in start_state]).long().to(device)
    rows = torch.Tensor([i // 9 for i in range(81)]).long().to(device)
    columns = torch.Tensor([i % 9 for i in range(81)]).long().to(device)
    X = torch.cat([embed(start_state, EMB_SIZE), embed(rows, EMB_SIZE), embed(columns, EMB_SIZE)], dim=1).float().to(device)
     
    return X
    
def get_start_embeds_batched(embed, start_state_batched):
    return torch.stack([get_start_embeds(embed, start_state) for start_state in start_state_batched])

def message_passing(nodes, edges, message_fn):
    n_nodes = nodes.shape[0]
    n_edges = edges.shape[0]
    n_embed = nodes.shape[1]

    message_inputs = nodes[edges]
    message_inputs = message_inputs.view(n_edges, 2*n_embed)
    messages = message_fn(message_inputs)

    updates = torch.zeros(n_nodes, n_embed).to(device)
    idx_j = edges[:, 1].to(device)
    updates = updates.index_add(0, idx_j, messages)
    return updates

def message_passing_batched(nodes_batched, edges, message_fn):
    return torch.stack([message_passing(nodes, edges, message_fn) for nodes in nodes_batched])

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_out = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc(x))
        x = F.relu(self.fc(x))
        x = self.fc_out(x)
        return x

class Pred(nn.Module):
    def __init__(self, input_size):
        super(Pred, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
    def forward(self, x):
        x = self.fc1(x)
        return x

def one_hot(num):
    return torch.Tensor

traindataset = MyDataset('train.csv')
trainloader = DataLoader(traindataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

mlp1 = MLP(3*EMB_SIZE).to(device)
mlp2 = MLP(2*HIDDEN_SIZE).to(device)
mlp3 = MLP(2*HIDDEN_SIZE).to(device)
r = Pred(HIDDEN_SIZE).to(device)
lstm = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True).to(device)
embed = torch.nn.functional.one_hot


optimizer_mlp1 = torch.optim.Adam(mlp1.parameters(), lr=2e-4, weight_decay=1e-4)
optimizer_mlp2 = torch.optim.Adam(mlp2.parameters(), lr=2e-4, weight_decay=1e-4)
optimizer_mlp3 = torch.optim.Adam(mlp3.parameters(), lr=2e-4, weight_decay=1e-4)
optimizer_r = torch.optim.Adam(r.parameters(), lr=2e-4, weight_decay=1e-4)
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=2e-4, weight_decay=1e-4)

optimizers = [optimizer_mlp1, optimizer_mlp2, optimizer_mlp3, optimizer_r, optimizer_lstm]
criterion = nn.CrossEntropyLoss()

edges = get_edges()

start_time = time.time()
for epoch in range(1000):
    for batch_id, start_state_batched in enumerate(trainloader):
        Y = start_state_batched[:, 1, :].to(device)
        X = start_state_batched[:, 0, :].to(device)

        X = get_start_embeds_batched(embed, X)
        X = mlp1(X)
        H = X.detach().clone().to(device)


        for optimizer in optimizers:
            optimizer.zero_grad()
        
        loss = 0
        S = torch.zeros(1, X.shape[0], HIDDEN_SIZE).to(device)
        for i in range(32):
            H = message_passing_batched(H, edges, mlp2) # message_fn
            H = mlp3(torch.cat([H, X], dim=2))
            H, S = lstm(H)
            res = r(H)
            
            pred = res
            for j in range(Y.shape[0]):
                # print(loss)
                loss += criterion(pred[j], Y[j])
                # print("step ", i, ": ",loss)
            # print(loss)
        loss /= Y.shape[0]
        if(batch_id % 10 == 0):
            print("10 batches time:", time.time() - start_time)
            start_time = time.time()
        print(batch_id, '/', len(trainloader))
        print(loss)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()





