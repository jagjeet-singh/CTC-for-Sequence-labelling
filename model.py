#from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from CTC import CTC
from math import floor
import matplotlib.pyplot as plt
import matplotlib
import pdb
import time

use_cuda = torch.cuda.is_available()

labels = torch.Tensor(np.load('dataset/labels.npy')).int()
data = torch.Tensor(np.load('dataset/data.npy'))
len = data.shape[2]

if len<1600:
    diff = 1600-len
    diff_seq = torch.zeros((data.shape[0],data.shape[1],diff))
    data = torch.cat((data,diff_seq), 2)

if len>1600:
    data = data[:,:,:1600]

if use_cuda:
    labels = labels.cuda()
    data = data.cuda()

class CTCNet(nn.Module):

    def __init__(self, hidden_size=180, num_seq=200):
        super(CTCNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.num_seq = num_seq
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.fc1 = nn.Linear(in_features=36000, out_features=18000)
        self.fc2 = nn.Linear(in_features=18000, out_features=9000)
        self.fc3 = nn.Linear(in_features=9000, out_features=2200)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros((2,1,self.hidden_size))),
            Variable(torch.zeros((2,1,self.hidden_size))))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        o,h = self.lstm(x.view(self.num_seq,1,self.hidden_size), self.hidden)
        x = o.view(self.num_seq*self.hidden_size)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(-1,11)
        out = F.softmax(x)
        return out

def editDistDP(str1, str2, m, n):

    str1 = str1.reshape((m))
    str2 = str2.reshape((n))
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
 
    # Fill d[][] in bottom up manner
    for i in range(m+1):
        for j in range(n+1):
            
            if i == 0:
                dp[i][j] = j    
 
            elif j == 0:
                dp[i][j] = i    
 
          
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
    return dp[m][n]



# Calculating hidden_size assuming fixed architecture of conv layers
h = data.shape[1]
w = data.shape[2]
num_seq = 200
hidden_size = 180

CTCloss = CTC()
net = CTCNet(hidden_size,num_seq)
if use_cuda:
    net = net.cuda()
optimizer = optim.SGD(net.parameters(), lr = 0.001)
train_size = labels.shape[0]
seq_width = data.shape[2]
num_classes = 11
#loss_arr = np.zeros((3))
#editDistance = np.zeros((3))
loss_arr = np.zeros((30*train_size))
editDistance = np.zeros((30*train_size))
start = time.time()
for epoch in range(15):
    for j in range(train_size):

        seq = Variable(data[j,:,:])
        label_seq = Variable(labels[j,:],requires_grad=False)
        if use_cuda:
            seq = seq.cuda()
            label_seq = label_seq.cuda()
        optimizer.zero_grad()

        output = net(seq.view(1,1,h,w))
        y_pred = torch.t(output)
        [loss, alphas] = CTCloss.forward(y_pred.data, label_seq.data)
        grad = CTCloss.backward(torch.Tensor(alphas), y_pred.data, label_seq.data)
        y_pred.backward(torch.Tensor(grad))
        optimizer.step()


        # Removing blanks 
        [pred_prob,pred_label] = torch.max(y_pred, 0)
        seq_without_blanks = pred_label[pred_label != 0]
        result = seq_without_blanks.data.numpy()-1
        result = result.reshape(1,np.size(result))
        #loss_arr[j] = loss
        loss_arr[train_size*epoch+j] = loss
        #pdb.set_trace()
        #editDistance[j] = editDistDP(result, label_seq.data.numpy(), result.shape[1], label_seq.size(0))
        ed = editDistDP(result, label_seq.data.numpy(), result.shape[1], label_seq.size(0))
        editDistance[train_size*epoch+j] = ed

        #pdb.set_trace()
        if j%10==0:
            print str(epoch+1) + '/' + str(30) + ' : ' + str(j+1)+'/'+str(train_size) + ' : ' + str(result.shape) + ' : ' +str(loss) + ' : ' + str(ed) + ':' + 'Time ' +str(time.time()-start)
            #print str(epoch)+' : '+str(j)+' : '+ str(result.shape)+' : '+str(loss)
            print result
            
    print('Saving model')
    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    },'TrainedModel_epoch' + str(epoch+1) +'.pt')       


np.savetxt('loss.txt', loss_arr)
np.savetxt('editDistance.txt', editDistance)

x_axis=range(3)

fig, ax=plt.subplots()
ax.plot(x_axis,loss_arr,'-b.',label='Training Loss vs iterations')
plt.ylabel('Loss')
plt.xlabel('iteration')
plt.title('Training loss')
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
legend = ax.legend(loc='upper center', shadow=True)
plt.savefig('loss.png')
#plt.show()

fig, ax=plt.subplots()
ax.plot(x_axis,editDistance,'-b.',label='Edit Distance vs iterations')
plt.ylabel('Edit Distance')
plt.xlabel('iteration')
plt.title('Training Edit Distance')
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
legend = ax.legend(loc='upper center', shadow=True)
plt.savefig('editDist.png')
#plt.show()
