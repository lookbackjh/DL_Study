
class Trainer():

    def __init__(self, model, dataloader, optimizer, loss_fn,device):
        # model, dataloader, optimizer, loss_fn
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device=device
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for (x, y) in self.dataloader:
                # device 설멎ㅇ
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass

                y_pred = self.model(x)
                
                # Compute Loss y의 차원을 맞추기 위해 squeeze를 사용
                y_pred = y_pred.squeeze()
                loss = self.loss_fn(y_pred, y)
                self.optimizer.zero_grad() # zero grad를 해주어야 새로운 grad를 계산할 수 있다.
                loss.backward() # Backward pass-> gradient 계산
                self.optimizer.step() # Update weights loss에 걸려있는 모든 weight를 업데이트한다.


            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

    def test(self,test_dataloader):
        self.model.eval() # Set the model to evaluation mode
        for (x, y) in test_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(x) # Forward pass
            y_pred = y_pred.squeeze()
            loss = self.loss_fn(y_pred, y)
            print("Test Loss: {}".format(loss.item()))