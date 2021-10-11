import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import random


def main():
    
    start_time = time.process_time() 

    torch.cuda.empty_cache()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

## read in all arguemnts
    par = argparse.ArgumentParser()
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    arg('-st' ,'--st' , type=str  , help='define the scantype')
    arg('-det','--det', type=str  , help='define the detector') 
 
    arg('--lr', type=float, help='X learning rate, default 1e-4')
    arg('--bs', type=int, help='X batch size, default 64')
    arg('--epochs', type=int, help='X number of epoches, default 15')
    
    args = par.parse_args()
    argsv = vars(par.parse_args())
    print(argsv)

    det = argsv['det']
    st = argsv['st']

    #del model 
    #del loss_fn
    #del optimizer


    output_dir = "/lfs/l1/legend/users/kilgus/MachineLearning/"
    output_name = output_dir + det + "_" + st + ".pdf"
    pdf = PdfPages(output_name)

    if argsv['lr']: 
        learning_rate = argsv['lr']
    else:
        learning_rate = 0.001

    if argsv['bs']: 
        batch_size = argsv['bs']
    else:
        batch_size = 64

    if argsv['epochs']: 
        epochs = argsv['epochs']
    else:
        epochs = 20

    print(learning_rate, epochs)

#Setting reproducability
    '''manualSeed = 7

    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)


    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True'''


## read in data an create datasets for training and testing 
    train_dataloader, test_dataloader = create_datasets(batch_size, argsv['st'], argsv['det'], output = pdf)

##select model, loss and optimizer
    model = ANN()
    #model = RNN(X_size=50, output_size=1, hidden_dim=25, n_layers=6, device=device)
    model.to(device)

    #loss_fn = nn.MSELoss()
    #loss_fn = nn.NLLLoss()
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss = np.array([])

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = np.append(loss, [train_loop(train_dataloader, model, loss_fn, optimizer, device, output = pdf)])
        test_loop(test_dataloader, model, loss_fn, device, output = pdf)
    print("Done!")

    plt.figure(figsize=(14,7))
    plt.plot(loss)
    plt.xlabel("epoche*batch")
    plt.ylabel("loss")
    pdf.savefig()
    plt.close()

## measure calculation time
    end_time = time.process_time() 
    print("Process time", end_time - start_time, "seconds")

    #model.close()

    pdf.close()

    del model
    del loss_fn
    del loss
    del optimizer

    torch.cuda.empty_cache()


    exit()





def create_datasets(batch_size, st, det, output = None): 

    #read in test and training data
    X_dir = "/lfs/l1/legend/users/kilgus/MachineLearning/output/"
    X_file = X_dir + det + "_" + st + ".npy"

    data = np.load(X_file)
    #data2 = np.load("/local/scratch0/astro/kilgus/MachineLearning/C00ANG2_th_HS2_top_psa.npy")
    data2 = np.empty((0,2), float)
    #print(data.shape)

    '''with open('/lfs/l1/legend/users/kilgus/MachineLearning/run113.json') as json_file: 
        data = json.load(json_file)
        #for d in data:
        y = np.array(data['flag'])
        X = np.array(data['weights'])
        #ID = np.array(data['id'])
        #y = np.array(data['flag'])'''
    #print(X.shape, y.shape)

    #X = X[np.where(ID==5)[0]]
    #y = y[np.where(ID==5)[0]]

    X = data[:,1:]
    X = np.append(X,data2[:,1:])
    y = data[:,0]
    y = np.append(y,data2[:,0])
    print(X)
    print(y)
    X = np.reshape(X, (-1,50))

    t = np.arange(1, 100, 2)


    a = np.where(y > 1)
    if len(a[0]) != 0: 
        print("Something went wrong with your dataset labels. Please check again.")
        exit()

    print(len(y))
    size_trainingset = int(len(y)*0.8)
    size_testset = len(y)-size_trainingset

    size_trainingset = 3000
    size_testset = 1000

    training_X = X[:size_trainingset]
    training_y = y[:size_trainingset]
    test_X = X[size_trainingset:size_trainingset+size_testset]
    test_y = y[size_trainingset:size_trainingset+size_testset]

    tensor_train_X = torch.Tensor(training_X) 
    tensor_train_y = torch.Tensor(training_y)
    train_dataset = TensorDataset(tensor_train_X,tensor_train_y) # create your datset
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size) # create your dataloader

    tensor_test_X = torch.Tensor(test_X) 
    tensor_test_y = torch.Tensor(test_y)
    test_dataset = TensorDataset(tensor_test_X,tensor_test_y) # create your datset
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size) # create your dataloader

    figure = plt.figure(figsize=(14, 14))
    cols, rows = 4, 8
    print(X.shape)
    print(y.shape)

    mse = np.where(y == 0)[0]
    sse = np.where(y == 1)[0]

    #print(len(X[sample_idx,:]))
    for i in range(1, (cols * rows + 1)):
        
        #img = X[sample_idx]
        figure.add_subplot(rows, cols, i)
        #plt.axis("off")
        #plt.title("label {}".format(y[mse[i]]))
        #plt.imshow(img.squeeze(), cmap="gray")
        plt.plot(X[mse[i],:], np.arange(1,100,2))
        plt.plot(X[sse[i],:], np.arange(1,100,2))

    output.savefig()
    plt.close()



    return train_dataloader, test_dataloader




def train_loop(dataloader, model, loss_fn, optimizer, device, output = None):
    size = len(dataloader.dataset)
    #print(size)

    model.to(device)
    
    #idx = torch.randperm(size)

    #plt.figure(figsize=(14,7))
    loss_print = np.array([])
    
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

    
        
        # Compute prediction and loss for Adam optimizer
        optimizer.zero_grad()
        pred = model(X)                                    ##for ANN
        #pred, _ = model(X)                                  ##for RNN
        #pred = torch.reshape(pred, (-1,))
        #print("pred = ", pred.shape, "y = ", y.shape)
        pred = torch.reshape(pred, (-1,))
        #print("pred = ", pred.shape, "y = ", y.shape)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        '''

        # Compute prediction and loss for LBFGS optimizer
        def closure():
            optimizer.zero_grad()
            pred = model(X)
            #pred, _ = model(X)
            pred = torch.reshape(pred, (-1,))
            #print("pred = ", pred.shape)
            loss = loss_fn(pred, y)
            loss.backward()

            return loss
        optimizer.step(closure)
        '''
        

        #l = closure().cpu()                             #for LBFGS
        l = loss.cpu()                                 #for Adam
        l = l.detach().numpy()
        loss_print = np.append(loss_print, l)
        #print(loss)



        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)                        # for Adam
            #loss, current = closure().item(), batch * len(X)                    # for LBFGS
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    #plt.plot(loss)
    #plt.xlabel("epoche*batch")
    #plt.ylabel("loss")
    #output.savefig()
    #plt.close()

    return loss_print


def test_loop(dataloader, model, loss_fn, device, output = None):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    a = 0
    p = np.array([])
    label = np.array([])

    plt.figure(figsize=(14,7))

    model.to(device)
    mse, sse = 0, 0
    
    with torch.no_grad():
        #for X, y in dataloader:
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)                    ##for ANN
            #pred, _ = model(X)                  ##for RNN

            pred = torch.reshape(pred, (-1,))
            #y = y.long()
            test_loss += loss_fn(pred, y).item()
            pred = pred.cpu()
            y = y.cpu()


            #print(pred)
            correct += (np.round_(pred) == y).type(torch.float).sum().item()

            mse += len(np.where(pred < 0.3)[0])
            sse += len(np.where(pred > 0.3)[0])

            if a == 0:
                print_image(X, pred, y, output)
                a += 1

            pred = pred.detach().numpy()
            y = y.detach().numpy()
            p = np.append(p, pred)
            label = np.append(label, y)

    
    p = np.reshape(p, (-1,))
    label = np.reshape(label, (-1,))
    #print("pred = ", p)
    plt.hist(p[np.where(label == 1)[0]], bins=100, color='red', hatch='\\', histtype='step')
    plt.hist(p[np.where(label == 0)[0]], bins=100, color='blue', hatch='/', histtype='step')
    output.savefig()
    plt.close()

    print(sse, mse)
    fraction = sse/(sse + mse)
    print(f"Survival fraction in DEP: {fraction*100}% \n")
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def print_image(X, pred, y, output):
    
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 2, 4
    print(X.shape)
    print(y.shape)
    X = X.cpu()
    #print(len(X[sample_idx,:]))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(y), size=(1,)).item()
        #img = X[sample_idx]
        figure.add_subplot(rows, cols, i)
        #plt.axis("off")
        plt.title("prediction {} \n label {}".format(pred[sample_idx], y[sample_idx]))
        #plt.imshow(img.squeeze(), cmap="gray")
        plt.plot(X[sample_idx,:], np.arange(1, 100, 2))
    output.savefig()
    plt.close()


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(50, 70),
            #nn.Sigmoid(), #or 
            nn.Tanh(),
            nn.Linear(70, 150),
            #nn.Sigmoid(), #or 
            nn.Tanh(),
            nn.Linear(150, 90),
            #nn.Sigmoid(), #or 
            nn.Tanh(),
            nn.Linear(90, 64),
            #nn.Sigmoid(), #or
            nn.Tanh(),
            nn.Linear(64, 34),
            #nn.Sigmoid(), #or 
            nn.Tanh(),
            nn.Linear(34, 10),
            #nn.Sigmoid(), #or 
            nn.Tanh(),
            nn.Linear(10, 1),
            nn.Sigmoid() #or 
            #nn.Tanh()
        )
        self.original = nn.Sequential(
            nn.Linear(50, 51),
            nn.Sigmoid(), 
            nn.Linear(51, 50),
            nn.Sigmoid(), #or 
            nn.Linear(50, 1),
            nn.Sigmoid() #or 
            #nn.Tanh()
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = self.original(x)
        #x = self.sigmoid(x)

        return x  




class RNN(nn.Module):
    def __init__(self, X_size, output_size, hidden_dim, n_layers, device):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.X_size = X_size
        self.n = 1
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.device = device

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(X_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first X using method defined below
        hidden = self.init_hidden(batch_size)   
        hidden = hidden.to('cuda:0') 

        x = torch.reshape(x,(batch_size, 1, self.X_size))

        # Passing in the X and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden 




if __name__=="__main__":
    main()
