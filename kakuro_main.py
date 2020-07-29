import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import sys
import json
import random

# sys.stdout = open('lologi.txt', 'w')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
EMB_SIZE = 40 # bedzie trzeba to zmienic jezeli bede chcial wejsc w wieksze kakuro
HIDDEN_SIZE = 96
BATCH_SIZE = 16


################################# Tutaj dodaje nowe rzeczy #################################

class Loader():

    def __init__(self, filename):
        self.dataset = []
        self.id = 0
        with open(filename, 'r') as file:
            for line in file:
                instance = json.loads(line)
                self.dataset.append(instance)
    
    def __len__(self):
        return len(self.dataset) // BATCH_SIZE
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.id + BATCH_SIZE >= len(self.dataset):
            random.shuffle(self.dataset)
            self.id = 0
 
        b = 0
        X = []
        Y = []
        edges = []
        L = []

        for idx in range(BATCH_SIZE):
            gen_x = [x[1] for x in self.dataset[self.id + idx][0]]
            gen_y = [y[1] for y in self.dataset[self.id + idx][1]]
            gen_edges = [(i + b, j + b) for i, j in self.dataset[self.id + idx][2]]
            b += len(gen_x)
            X += gen_x
            Y += gen_y
            edges += gen_edges
            L.append(len(gen_x))
            
        self.id += BATCH_SIZE
        return torch.Tensor(X).long(), torch.Tensor(Y).long(), torch.Tensor(edges).long(), torch.Tensor(L).long()



def get_start_embeds(embed, X):
    return embed(X, EMB_SIZE).float()


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


trainloader = Loader("../Kakurosy/train_prep.txt")
testloader = Loader("../Kakurosy/val_prep.txt")


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

start_time = time.time()

def check_val():
    with torch.no_grad():
        almost_correct = 0
        correct = 0
        total = 0
        for batch_id, (X, Y, E, L) in enumerate(testloader):
            if(batch_id // len(testloader)):
                break
            X = get_start_embeds(embed, X)
            X = X.to(device)
            Y = Y.to(device)

            X = mlp1(X)
            H = X.detach().clone().to(device)

            CellState = torch.zeros(X.shape).to(device)
            HiddenState = torch.zeros(X.shape).to(device)
            for i in range(32):
                H = message_passing(H, E, mlp2) # message_fn
                H = mlp3(torch.cat([H, X], dim=1))
                HiddenState, CellState = lstm(H, (HiddenState, CellState))
                H = CellState
                pred = r(H)


            pred = torch.argmax(pred, dim=1)
            amam = (pred == Y)
            
            current_id = 0
            correct_batch = torch.zeros((L.shape[0])).long()
            for idx, l in enumerate(L):
                guessed_right = torch.sum(amam[current_id:(current_id+l)])
                correct_batch[idx] = guessed_right
                current_id += l

            correct += torch.sum(torch.sum(correct_batch == L))
            almost_correct += torch.sum(torch.sum(correct_batch > L* 0.6))
            total += BATCH_SIZE
        
        print("Correctly solved: {}, out of: {}".format(correct, total))
        print("Almost correctly solved: {}, out of: {}".format(almost_correct, total))


running_loss = 0
for batch_id, (X, Y, E, _) in enumerate(trainloader, 1):
    epoch = batch_id  // len(trainloader)
    is_new_epoch = batch_id // len(trainloader) > (batch_id-1)  // len(trainloader)
    
    X = get_start_embeds(embed, X)
    X = X.to(device)
    Y = Y.to(device)
    E = E.to(device)

    X = mlp1(X)
    H = X.detach().clone().to(device)


    for optimizer in optimizers:
        optimizer.zero_grad()
    
    loss = 0
    CellState = torch.zeros(X.shape).to(device)
    HiddenState = torch.zeros(X.shape).to(device)
    for i in range(32):
        H = message_passing(H, E, mlp2) # message_fn
        H = mlp3(torch.cat([H, X], dim=1))
        HiddenState, CellState = lstm(H, (HiddenState, CellState))
        H = CellState
        pred = r(H)
        
        loss += criterion(pred, Y)
    
    loss /= BATCH_SIZE
    running_loss += loss
    const = 100
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
    
    if is_new_epoch and epoch % 4 == 1:
        check_val()