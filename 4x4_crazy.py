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
# device = torch.device("cpu")
CUDA_ID = 1
EMB_SIZE = 50
HIDDEN_SIZE = 96
BATCH_SIZE = 1
LEARNING_RATE = 2e-4 
DEBUG = False
NUM_STEPS = 32
PATH_CHECKPOINT = "./kakuro_checkpoint_4x4"
TEST_PATH = "../Kakurosy/ready_datasets/4x4_expert_small.txt"
device = torch.device("cuda:{}".format(CUDA_ID) if torch.cuda.is_available() else "cpu")

################################# Tutaj dodaje nowe rzeczy #################################

class Loader():

    def __init__(self, filename):
        self.dataset = []
        self.id = 0
        with open(filename, 'r') as file:
            for line in file:
                # instance = json.loads(line)
                self.dataset.append(line)
        self.dataset = [json.loads(line) for line in set(self.dataset)]
        print("number of examples: ", len(self.dataset))
    
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


def save_checkpoint(networks):
    torch.save({
        'mlp1': networks[0].state_dict(),
        'mlp2': networks[1].state_dict(),
        'mlp3': networks[2].state_dict(),
        'r': networks[3].state_dict(),
        'lstm': networks[4].state_dict()
    }, PATH_CHECKPOINT)

def load_checkpoint(networks):
    checkpoint = torch.load(PATH_CHECKPOINT)
    networks[0].load_state_dict(checkpoint['mlp1'])
    networks[1].load_state_dict(checkpoint['mlp2'])
    networks[2].load_state_dict(checkpoint['mlp3'])
    networks[3].load_state_dict(checkpoint['r'])
    networks[4].load_state_dict(checkpoint['lstm'])

if __name__ == '__main__':
    testloader = Loader(TEST_PATH)

    mlp1 = MLP(EMB_SIZE).to(device)
    mlp2 = MLP(2*HIDDEN_SIZE).to(device)
    mlp3 = MLP(2*HIDDEN_SIZE).to(device)
    r = Pred(HIDDEN_SIZE).to(device)
    lstm = nn.LSTMCell(HIDDEN_SIZE, HIDDEN_SIZE).to(device)
    embed = torch.nn.functional.one_hot
    networks = [mlp1, mlp2, mlp3, r, lstm]
    load_checkpoint(networks)
    for net in networks:
        net.eval()
    correct = 0
    correct_cell = 0
    total = 0
    total_cell = 0
    
    for batch_id, (X, Y, E, L) in enumerate(testloader):
        if(batch_id // len(testloader)):
                break
        X = get_start_embeds(embed, X)
        X = X.to(device)
        Y = Y.to(device)
        E = E.to(device)

        X = mlp1(X)
        H = X.detach().clone().to(device)

        CellState = torch.zeros(X.shape).to(device)
        HiddenState = torch.zeros(X.shape).to(device)
        for i in range(NUM_STEPS):
            H = message_passing(H, E, mlp2) # message_fn
            H = mlp3(torch.cat([H, X], dim=1))
            HiddenState, CellState = lstm(H, (HiddenState, CellState))
            H = CellState
            pred = r(H)

        pred = torch.argmax(pred, dim=1)
        # print(pred)
        amam = (pred == Y)
        correct_cell += torch.sum(amam)
        total_cell += len(amam)

        current_id = 0
        correct_batch = torch.zeros((L.shape[0])).long()
        for idx, l in enumerate(L):
            guessed_right = torch.sum(amam[current_id:(current_id+l)])
            correct_batch[idx] = guessed_right
            current_id += l

        correct += torch.sum(torch.sum(correct_batch == L))
        total += BATCH_SIZE

        
    test_acc = float(correct) / total
    test_acc_cell = float(correct_cell) / total_cell

    print("correct: ", correct, " | total: ", total)

    print("accuracy: {:.4f}".format(test_acc))
    print("cell accuracy: {:.4f}".format(test_acc_cell))