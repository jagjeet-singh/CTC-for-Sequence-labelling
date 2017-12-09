#from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from starter2 import CTC
from math import floor
import matplotlib.pyplot as plt
import pdb
#import visualize


labels = torch.Tensor(np.load('dataset/labels_10seq.npy')).int()
data = torch.Tensor(np.load('dataset/data_10seq.npy'))

class CTCNet(nn.Module):

    def __init__(self):
        super(CTCNet, self).__init__()
        self.hidden_size = 50
        self.num_layers = 1
        # 36x10 Input and 36x10x12 output
        self.conv = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 3, padding = 1)
        # 4320x1 Input and 50x1 output
        self.lstm = nn.LSTM(input_size = 1080, hidden_size = 50, num_layers = 1)
        # 50 to 11 MLP
        self.fc = nn.Linear(in_features =50, out_features = 11)
        self.softmax = nn.Softmax()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.ones((1,1,self.hidden_size))*.1),
            Variable(torch.ones((1,1,self.hidden_size))*.1))
        #return (Variable(torch.zeros(1, 1, self.hidden_size)),
         #   Variable(torch.zeros(1, 1, self.hidden_size)))

    def forward(self, x): 
        x = F.relu(self.conv(x))
        x = x.view(1,1,1080)
        x,y  = self.lstm(x,self.hidden)
        x = F.relu(self.fc(x))
        x = x.view(1,11)
        x = self.softmax(x)
        return x

# Training starts ####

def split_seq_stride_1(seq):
     split_width = 10
     stride = 10
     seq_width = seq.shape[1]
     num_splits = seq_width-stride
     splits = Variable(torch.Tensor(np.zeros((seq.shape[0], split_width, num_splits))), requires_grad=False)
     for i in range(num_splits):
         splits.data[:,:,i] = seq[:,i:i+10]
     return splits

def split_seq(seq):
    split_width = 10
    stride = 10
    seq_width = seq.shape[1]
    num_splits = int(seq_width/split_width)
    splits = Variable(torch.Tensor(np.zeros((seq.shape[0], split_width, num_splits))))
    for i in range(num_splits):
        splits.data[:,:,i] = seq[:,10*i:10*i+10]
    return splits

CTCloss = CTC()
net = CTCNet()
optimizer = optim.RMSprop(net.parameters(), lr = 0.1)
train_size = labels.shape[0]
seq_width = data.shape[2]
num_classes = 11

for epoch in range(100):
    for j in range(train_size):
        #pdb.set_trace()
        seq = data[j,:,:]
        label_seq = Variable(labels[j,:],requires_grad=False )
        #label_seq = labels[j,:] 
        splits = split_seq(seq)
        num_splits = splits.data.shape[2]
        #y_pred = Variable(torch.Tensor(np.zeros((num_classes,num_splits))))
        #print num_splits
	#pdb.set_trace()
        for i in range(num_splits):
            input_split = splits[:,:,i]
            input_split = input_split.contiguous().view(1,1,input_split.data.shape[0], input_split.data.shape[1])
            optimizer.zero_grad()
            output = net(input_split)
	    if i==0:
		y_pred = output.view(11,1)
	    else:
            	y_pred = torch.cat((y_pred, output.view(11,1)),1)
		#y_pred[:,i] = output
        #print '########### Sequence completed ##########'
	
	#y_pred.register_hook(print)
        #pdb.set_trace()
        [loss, alphas] = CTCloss.forward(y_pred.data, label_seq.data)
        grad = CTCloss.backward(torch.Tensor(alphas), y_pred.data, label_seq.data)
        y_pred.backward(torch.Tensor(grad))
        optimizer.step()

        # Removing blanks 
        [pred_prob,pred_label] = torch.max(y_pred, 0)
        seq_without_blanks = pred_label[pred_label != 0]
        result = seq_without_blanks.data.numpy()-1
        result = result.reshape(1,np.size(result))

        #pdb.set_trace(
        print str(epoch)+' : '+str(j)+' : '+ str(result.shape)+' : '+str(loss)
	if j%100==0:
		print result
        #st = np.array([['### Line Changed ###']])
        #with open('results.txt','a') as f_handle:
         #       np.savetxt(f_handle, result, newline='\n', fmt='%-4d')
          #      np.savetxt(f_handle, st, newline='\n', fmt='%s')
	#visualize.make_dot(y_pred, params=None)

