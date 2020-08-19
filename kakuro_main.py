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
import neptune

# sys.stdout = open('lologi.txt', 'w')
# device = torch.device("cpu")
CUDA_ID = 0
EMB_SIZE = 50
HIDDEN_SIZE = 96
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
DEBUG = False
NUM_STEPS = 64

device = torch.device("cuda:{}".format(CUDA_ID) if torch.cuda.is_available() else "cpu")
neptune.init('andrzejzdobywca/GNN')
random.seed(3)

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
        random.shuffle(self.dataset)
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

def check_val():
    with torch.no_grad():
        almost_correct = 0
        correct = 0
        correct_cell = 0
        total_cell = 0
        total = 0
        running_loss = 0
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
            for i in range(NUM_STEPS):
                H = message_passing(H, E, mlp2) # message_fn
                H = mlp3(torch.cat([H, X], dim=1))
                HiddenState, CellState = lstm(H, (HiddenState, CellState))
                H = CellState
                pred = r(H)
                running_loss += criterion(pred, Y)
            

            pred = torch.argmax(pred, dim=1)
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
            almost_correct += torch.sum(torch.sum(correct_batch > L* 0.8))
            total += BATCH_SIZE
        
        val_loss = running_loss.item() / len(testloader)
        val_acc = float(correct) / total
        val_acc_cell = float(correct_cell) / total_cell
        print("Correctly solved: {}, out of: {}".format(correct, total))
        print("Almost correctly solved: {}, out of: {}".format(almost_correct, total))

        return val_loss, val_acc, val_acc_cell


def save_checkpoint(networks, pathcheckpoint):
    torch.save({
        'mlp1': networks[0].state_dict(),
        'mlp2': networks[1].state_dict(),
        'mlp3': networks[2].state_dict(),
        'r': networks[3].state_dict(),
        'lstm': networks[4].state_dict()
    }, pathcheckpoint)

def load_checkpoint(networks, pathcheckpoint):
    checkpoint = torch.load(patchckeckpoint)
    networks[0].load_state_dict(checkpoint['mlp1'])
    networks[1].load_state_dict(checkpoint['mlp2'])
    networks[2].load_state_dict(checkpoint['mlp3'])
    networks[3].load_state_dict(checkpoint['r'])
    networks[4].load_state_dict(checkpoint['lstm'])

if __name__ == '__main__':
    with neptune.create_experiment(params={'lr': LEARNING_RATE}) as exp:
        path_checkpoint = "./checkpoints/{}".format(exp.id)
        
        trainloader = Loader("../Kakurosy/train_{}_{}.txt".format(sys.argv[1], sys.argv[2]))
        testloader = Loader("../Kakurosy/val_{}_{}.txt".format(sys.argv[1], sys.argv[2]))

        mlp1 = MLP(EMB_SIZE).to(device)
        mlp2 = MLP(2*HIDDEN_SIZE).to(device)
        mlp3 = MLP(2*HIDDEN_SIZE).to(device)
        r = Pred(HIDDEN_SIZE).to(device)
        lstm = nn.LSTMCell(HIDDEN_SIZE, HIDDEN_SIZE).to(device)
        embed = torch.nn.functional.one_hot
        networks = [mlp1, mlp2, mlp3, r, lstm]
        save_checkpoint(networks, path_checkpoint)
        optimizer_mlp1 = torch.optim.Adam(mlp1.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        optimizer_mlp2 = torch.optim.Adam(mlp2.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        optimizer_mlp3 = torch.optim.Adam(mlp3.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        optimizer_r = torch.optim.Adam(r.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        optimizers = [optimizer_mlp1, optimizer_mlp2, optimizer_mlp3, optimizer_r, optimizer_lstm]
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()

        running_loss = 0
        best_val = 0
        epoch = 0
        correct = 0
        correct_cell = 0
        total = 0
        total_cell = 0
        print("Training has started")
        for batch_id, (X, Y, E, L) in enumerate(trainloader, 1):
            if epoch >= 1000:
                break
            is_new_epoch = batch_id // len(trainloader) > epoch 
            epoch = batch_id  // len(trainloader)
            
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
            for i in range(NUM_STEPS):
                H = message_passing(H, E, mlp2) # message_fn
                H = mlp3(torch.cat([H, X], dim=1))
                HiddenState, CellState = lstm(H, (HiddenState, CellState))
                H = CellState
                pred = r(H)
                
                loss += criterion(pred, Y)
            pred = torch.argmax(pred, dim=1)
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
            running_loss += loss
            const = 100
            if(batch_id % const == 0 and DEBUG):
                print("trainset: {} / {}".format(batch_id % len(trainloader), len(trainloader)), end= " | ")
                print("{:.6f} updates/s".format( const / (time.time() - start_time)))
                start_time = time.time()
                sys.stdout.flush()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            
            if is_new_epoch:
                train_loss = running_loss.item() / len(trainloader) # multiply the loss by 100 to see it better
                train_acc = float(correct) / total
                train_acc_cell = float(correct_cell) / total_cell

                print("epoch: {}".format(epoch-1))
                print("correct: ", correct, " | total: ", total)
                val_loss, val_acc, val_acc_cell = check_val()
                print("train_loss: {:.6f}".format(train_loss))
                print("train_acc: {:.4f}".format(train_acc))
                print("train_acc_cell: {:.4f}".format(train_acc_cell))
                print("val_loss: {:.6f}".format(val_loss))
                print("val_acc: {:.4f}".format(val_acc))
                print("val_acc_cell: {:.4f}".format(val_acc_cell))
                neptune.send_metric('train_loss', train_loss)
                neptune.send_metric('train_acc', train_acc)
                neptune.send_metric('train_acc_cell', train_acc_cell)
                neptune.send_metric('val_loss', val_loss)
                neptune.send_metric('val_acc', val_acc)
                neptune.send_metric('val_acc_cell', val_acc_cell)
                if val_acc > best_val:
                    best_val = val_acc
                    save_checkpoint(networks, path_checkpoint)
                
                ## reset running variables
                running_loss = 0
                correct = 0
                correct_cell = 0
                total = 0
                total_cell = 0