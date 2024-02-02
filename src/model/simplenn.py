import torch.nn as nn
class SimpleNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):
        # x: batch_size x 57
        x = self.linear1(x)

        #question
        # batchsize x 5
        x = self.linear2(x)
        # x: batch_size x hidden_dim
        x = self.sigmoid(x)
        # x: batch_size x 1
        
        return x
    

## lightning version
# import pytorch_lightning as pl
# import torch

# class SimpleNN(pl.LightningModule):

#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(10, 5)
#         self.linear2 = nn.Linear(5, 1)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):

#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.sigmoid(x)
        
#         return x
    
#     def training_step(self, batch, batch_idx):

#         x, y = batch
#         logits = self(x)
#         loss = nn.BCELoss()(logits, y)
#         return loss
    
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.02)
    

