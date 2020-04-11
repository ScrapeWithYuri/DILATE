import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys

def create_synthetic_dataset(N, N_input,N_output,sigma):
    # N: number of samples in each split (train, test)
    # N_input: import of time steps in input series
    # N_output: import of time steps in output series
    # sigma: standard deviation of additional noise
    X = []
    breakpoints = []

    df = pd.read_csv(
        r'C:\\Users\\russi\PycharmProjects\DILATE-master\data\EURUSD_Candlestick_1_Hour_BID_01.01.2011-04.04.2020.csv',
        index_col = False)

    xdata = df['Close']

    for k in range(2*N):
        serie = np.array([ sigma*random.random() for i in range(N_input+N_output)])

        #q = random.randint(k,k+30000)
        q = k

        # normalize train and test data to zero mean and unit variance
        meanSeq = np.mean(xdata[q:N_input+q])
        stdSeq = np.std(xdata[q:N_input+q])

        for i in range(N_input+N_output):
            #serie[i] = (xdata[i+q] - meanSeq) / stdSeq
            serie[i] = xdata[i + q]

        i1 = random.randint(1,10)
        i2 = random.randint(10,18)
        #j1 = random.random()
        #j2 = random.random()
        interval = abs(i2-i1) + random.randint(-3,3)
        #serie[i1:i1+1] += j1
        #serie[i2:i2+1] += j2
        #serie[i2+interval:] += (j2-j1)
        X.append(serie)
        breakpoints.append(i2+interval)
    X = np.stack(X)
    breakpoints = np.array(breakpoints)
    return X[0:N,0:N_input], X[0:N, N_input:N_input+N_output], X[N:2*N,0:N_input], X[N:2*N, N_input:N_input+N_output],breakpoints[0:N], breakpoints[N:2*N]


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target, breakpoints):
        super(SyntheticDataset, self).__init__()  
        self.X_input = X_input
        self.X_target = X_target
        self.breakpoints = breakpoints
        
    def __len__(self):
        return (self.X_input).shape[0]

    def __getitem__(self, idx):
        return (self.X_input[idx,:,np.newaxis], self.X_target[idx,:,np.newaxis]  , self.breakpoints[idx])