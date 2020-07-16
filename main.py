import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EMB_SIZE = 12
HIDDEN_SIZE = 96
BATCH_SIZE = 4

class MyDataset(Dataset):

    def __init__(self, csv_file):
        self.csv = np.array(pd.read_csv(csv_file, sep=',', header=None))

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        res = self.csv[idx]
        res = torch.Tensor([[int(x) for x in sudoku] for sudoku in res]).long()
        return res

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

def get_start_embeds(embed, embed_dict, start_state):
    start_state = torch.Tensor([x.item() for x in start_state]).long().to(device)
    X = embed(start_state)
    return X
    
def get_start_embeds_batched(embed, embed_dict, start_state_batched):
    return torch.stack([get_start_embeds(embed, embed_dict, start_state) for start_state in start_state_batched])

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
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(2*EMB_SIZE, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_out = nn.Linear(HIDDEN_SIZE, EMB_SIZE)
    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc(x))
        x = F.relu(self.fc(x))
        x = self.fc_out(x)
        return x

class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.fc1 = nn.Linear(3*EMB_SIZE, EMB_SIZE)
    def forward(self, x):
        x = self.fc1(x)
        return x

class Pred(nn.Module):
    def __init__(self):
        super(Pred, self).__init__()
        self.fc1 = nn.Linear(EMB_SIZE, 10)
    def forward(self, x):
        x = self.fc1(x)
        return x

def one_hot(num):
    return torch.Tensor

traindataset = MyDataset('sample.csv')
trainloader = DataLoader(traindataset, batch_size = 4, shuffle = True, num_workers = 4)

embed = nn.Embedding(10, EMB_SIZE).to(device)
word_to_emb = {i: i for i in range(10)}
message_fn = MLP().to(device)
g = nn.LSTM(3*EMB_SIZE, EMB_SIZE, HIDDEN_SIZE, batch_first=True).to(device)
r = Pred().to(device)
optimizer_embed = torch.optim.Adam(embed.parameters(), lr=2e-4, weight_decay=1e-4)
optimizer_r = torch.optim.Adam(r.parameters(), lr=2e-4, weight_decay=1e-4)
optimizer_g = torch.optim.Adam(g.parameters(), lr=2e-4, weight_decay=1e-4)
optimizer_mlp = torch.optim.Adam(message_fn.parameters(), lr=2e-4, weight_decay=1e-4)
optimizers = [optimizer_embed, optimizer_r, optimizer_g, optimizer_mlp]

criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    for start_state_batched in trainloader:
        Y = start_state_batched[:, 1, :].to(device)
        X = start_state_batched[:, 0, :].to(device)

        X = get_start_embeds_batched(embed, word_to_emb, X)
        H = X.detach().clone().to(device)

        edges = get_edges()

        for optimizer in optimizers:
            optimizer.zero_grad()
        
        loss = 0
        S = torch.zeros(HIDDEN_SIZE, BATCH_SIZE, EMB_SIZE).to(device)
        S = (S, S)
        for i in range(32):
            M = message_passing_batched(H, edges, message_fn)
            res = torch.cat([H, X, M], dim=2)
            H, S = g(res, S)
            # print(S[0].shape, S[1].shape)
            # print(H.shape)
            res = r(H)
            # pred = nn.Softmax(dim=2)(res)
            pred = res
            for j in range(Y.shape[0]):
                # print(loss)
                loss += criterion(pred[j], Y[j])
            loss /= Y.shape[0]
        print(loss)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()





