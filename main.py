from src.data.spambase import Spam
import pandas as pd
import numpy as np
#from src.model import model
from src.model.simplenn import SimpleNN
from src.trainer import Trainer
from torch.utils.data import DataLoader, Dataset
import torch

def trainer():
    spam = Spam()
    Spam_train,Spam_test = spam.create_data() # X: train, Y: train_label, X: test

    # Create a dataloader
    train_dataloader = DataLoader(Spam_train, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(Spam_test, batch_size=3000, shuffle=True)
    # Create a model
    model = SimpleNN(input_dim=57, hidden_dim=10,output_dim= 1)
    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    # Create an optimizer
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Create a loss function
    loss_fn = torch.nn.BCELoss()
    # Create a trainer
    trainer = Trainer(model, train_dataloader, optimizer, loss_fn,device)
    # Train the model
    trainer.train(num_epochs=5)
    # Test the model
    trainer.test(test_dataloader)



if __name__ == '__main__':
    trainer()
