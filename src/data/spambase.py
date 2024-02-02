import pandas as pd
import numpy as np
# torch dataloader
from torch.utils.data import DataLoader, Dataset
import torch
#train_test_split
class SpamDataloader(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



class Spam:

    def __init__(self):
        self.train_dir = 'src/dataset/spambase' + '/spambase.data'

        pass

    def create_data(self):
        X = pd.read_csv(self.train_dir, sep=',', header=None)
        #data.dropna(axis=1, how='all', inplace=True)
        X=X.values
        Y=X[:,-1]
        X=X[:,:-1]
        Y = Y.astype(int)
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train = torch.tensor(X_train).float()
        Y_train = torch.tensor(Y_train).float()
        X_test = torch.tensor(X_test).float()
        Y_test = torch.tensor(Y_test).float()
        Spam_train = SpamDataloader(X_train, Y_train)
        Spam_test = SpamDataloader(X_test, Y_test)
        return Spam_train, Spam_test


