
import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import Module
import torch.nn as nn
from beam_utils import codebook_blockmatrix


    



class DKNN(nn.Module):
    def __init__(self):
        super(DKNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=False)
        self.fc = nn.Linear(128 * 8 * 4, 128)  # Flattened size based on your data

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x
    

        
def fit(model, train_loader, val_loader, opt, loss_fn, EPOCHS):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.long()
           
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            train_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
        train_loss /= batch_idx + 1
        train_acc /= batch_idx + 1
        model.eval()
        val_loss = 0
        val_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.long()  
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch)
            val_loss += loss.detach().item()
            val_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
        val_loss /= batch_idx + 1
        val_acc /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 10 == 0:
            print('Epoch : {}, Training loss = {:.2f}, Training Acc = {:.2f}, Validation loss = {:.2f}, Validation Acc = {:.2f}.'.format(epoch,train_loss,train_acc,val_loss,val_acc))
    return train_loss_hist, val_loss_hist

def eval_model(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    test_acc = 0
    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        var_X_batch = X_batch.float()
        var_y_batch = y_batch.long ()  
        output = model(var_X_batch)
        loss = loss_fn(output, var_y_batch)
        test_loss += loss.detach().item()
        test_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
    test_loss /= batch_idx + 1
    test_acc /= batch_idx + 1
    return test_loss, test_acc