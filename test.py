import torch.nn as nn
import torch
class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: batch_size x 10
        x = self.linear1(x)
        #question
        # batchsize x 5
        x = self.linear2(x)
        # x: batch_size x 1
        x = self.sigmoid(x)
        # x: batch_size x 1
        
        return x    
    
    def print_params(self):
        for param in self.parameters():
            print(param)

model = SimpleNN()
#model.print_params()
x=torch.randn(100,10) # batch_size x 10

print(model(x))