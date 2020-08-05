import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import sys

# sys.stdout = open('lologi.txt', 'w')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMB_SIZE = 16
HIDDEN_SIZE = 96
BATCH_SIZE = 32

class MyDataset(Dataset):

    def __init__(self, csv_file):
        self.csv = np.array(pd.read_csv(csv_file, sep=',', header=None))
        self.csv = torch.Tensor([[[int(x) for x in my_input] for my_input in problem] for problem in self.csv]).long()
    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        # res = torch.Tensor([[int(x) for x in sudoku] for sudoku in self.csv[idx]]).long()
        # return res
        return self.csv[idx][0], self.csv[idx][1]

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
    
    edges_base = list(set(rows + columns + squares))
    batched_edges = [(i + (b * 81), j + (b * 81)) for b in range(BATCH_SIZE) for i, j in edges_base]
    return torch.Tensor(batched_edges).long()

def get_start_embeds(embed, X):
    # rows = embed(torch.Tensor([i // 9 for i in range(81)]).long(), EMB_SIZE).repeat(X.shape[0] // 81, 1) # beznadziejne rozwiazanie !!!!
    # columns = embed(torch.Tensor([i % 9 for i in range(81)]).long(), EMB_SIZE).repeat(X.shape[0] // 81, 1) # beznadziejne rozwiazanie, tez !!
    # X = torch.cat([embed(X, EMB_SIZE), rows, columns], dim=1).float()
    X = embed(X, EMB_SIZE).float()
     
    return X


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

testdataset = MyDataset('test.csv')
testloader = DataLoader(testdataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

mlp1 = MLP(EMB_SIZE).to(device)
mlp2 = MLP(2*HIDDEN_SIZE).to(device)
mlp3 = MLP(2*HIDDEN_SIZE).to(device)
r = Pred(HIDDEN_SIZE).to(device)
lstm = nn.LSTMCell(HIDDEN_SIZE, HIDDEN_SIZE).to(device)
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

def check_val():
    with torch.no_grad():
        almost_correct = 0
        correct = 0
        total = 0
        my_dict = [0 for i in range(82)]
        for batch_id, (X_batched, Y_batched) in enumerate(testloader):
            if X_batched.shape[0] != BATCH_SIZE:
                continue
            X = X_batched.flatten()

            X = get_start_embeds(embed, X)
            X = X.to(device)
            Y_batched = Y_batched.to(device)
            X = mlp1(X)
            H = X.detach().clone().to(device)

            CellState = torch.zeros(X.shape).to(device)
            HiddenState = torch.zeros(X.shape).to(device)
            for i in range(32):
                H = message_passing(H, edges, mlp2) # message_fn
                H = mlp3(torch.cat([H, X], dim=1))
                HiddenState, CellState = lstm(H, (HiddenState, CellState))
                H = CellState
                pred = r(H)


            pred = torch.argmax(pred, dim=1)

            pred = pred.view(-1, 81)
            amam = torch.sum(pred == Y_batched, dim=1)
            for x in amam:
                my_dict[x.item()] += 1

            # if batch_id % 100 == 0:
            #     print("validation: ", batch_id, '/', len(testloader))

            # print(torch.sum(X != 0, dim=1))
            correct += torch.sum(torch.sum(pred == Y_batched, dim=1) == 81)
            almost_correct += torch.sum(torch.sum(pred == Y_batched, dim=1) >= 60)
            total += Y_batched.shape[0]
        
        for i, x in enumerate(my_dict):
            print(i, ": ", x)
        
        print("Correctly solved: {}, out of: {}".format(correct, total))
        print("Almost correctly solved: {}, out of: {}".format(almost_correct, total))


for epoch in range(1000):
    running_loss = 0
    print("Started epoch: ", epoch)
    for batch_id, (X, Y) in enumerate(trainloader):
        if X.shape[0] != BATCH_SIZE:
            continue
        X = X.flatten()
        Y = Y.flatten()

        X = get_start_embeds(embed, X)

        X = X.to(device)
        Y = Y.to(device)

        X = mlp1(X)
        H = X.detach().clone().to(device)


        for optimizer in optimizers:
            optimizer.zero_grad()
        
        loss = 0
        CellState = torch.zeros(X.shape).to(device)
        HiddenState = torch.zeros(X.shape).to(device)
        for i in range(32):
            H = message_passing(H, edges, mlp2) # message_fn
            H = mlp3(torch.cat([H, X], dim=1))
            HiddenState, CellState = lstm(H, (HiddenState, CellState))
            H = CellState
            pred = r(H)
            
            loss += criterion(pred, Y)
        
        loss /= BATCH_SIZE
        running_loss += loss
        const = 200
        if(batch_id % const == 0):
            print("trainset: {} / {}".format(batch_id, len(trainloader)), end= " | ")
            print("{:.6f} updates/s".format( const / (time.time() - start_time)), end=" | ")
            print("train_loss: {:.6f}".format(running_loss.item() / const))
            running_loss = 0
            sys.stdout.flush()
            start_time = time.time()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
    check_val()





