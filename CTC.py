import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb 
import math

#parser.add_argument('--batch-size', type=int, default=64, metavar='N',
 #                   help='input batch size for training (default: 64)')

#args = parser.parse_args()
# See http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

class CTC(torch.autograd.Function):

    def forward(self, y_pred, seq):

        # y_pred = nxm, where n = number of labels and m = no. of time frames
        # seq = sequence of labels
        # blank = position of blank
        # yi_pred is a Variable
        # seq is a variable

        y_pred = y_pred.numpy()
        seq = seq.numpy()
        #pdb.set_trace()
        blank = 0 
        L = seq.shape[0]
        numDigits = y_pred.shape[0]
        L_with_blanks = 2*L + 1 
        T = y_pred.shape[1]
        alphas = np.zeros((L_with_blanks,T))

   
        # Initialize alphas
        alphas[0,0] = y_pred[blank, 0]
        alphas[1,0] = y_pred[seq[0], 0]
        total_alpha = np.sum(alphas[:,0])
        alphas[:,0] = alphas[:,0]/total_alpha
        llForward = np.log(total_alpha) 

        #Forward Pass

        for t in xrange(1,T):
            start = max(0, L_with_blanks-2*(T-t))
            end = min(2*t+2, L_with_blanks)
            for s in xrange(start, L_with_blanks):
                l = (s-1)/2
                
                #blank
                if s%2 == 0:
                    if s == 0:
                        alphas[s,t] = alphas[s, t-1]*y_pred[blank,t]
                    else:
                        alphas[s,t] = (alphas[s,t-1]+alphas[s-1,t-1])*y_pred[blank,t]

                #same label twice

                elif s == 1 or seq[l] == seq[l-1]:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1])*y_pred[seq[l],t]
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1])*y_pred[seq[l],t]
            
        
            c = np.sum(alphas[start:end,t])
            alphas[start:end,t] = alphas[start:end,t] / c 
            llForward += np.log(c)

	return -llForward, alphas
	

    def backward(self, alphas, y_pred, seq):
	
	    y_pred = y_pred.numpy()
        seq = seq.numpy()
	    alphas = alphas.numpy()
        blank = 0
        L = seq.shape[0]
        numDigits = y_pred.shape[0]
        L_with_blanks = 2*L + 1
        T = y_pred.shape[1]
        ab = np.empty((L_with_blanks,T))
        betas = np.zeros((L_with_blanks,T))
        grad = np.zeros((numDigits,T))

        grad_v = grad
        betas[-1,-1] = y_pred[blank,-1]
        betas[-2,-1] = y_pred[seq[-1],-1]
        c = np.sum(betas[:,-1])
        betas[:,-1] = betas[:,-1]/c
        llBackward = np.log(c)

        for t in xrange(T-2,-1,-1):
                start = max(0, L_with_blanks-2*(T-t))
                end = min(2*t+2, L_with_blanks)
                for s in xrange(end-1,-1,-1):
                        l = (s-1)/2
                        # blank
                        if s%2 == 0:
                                if s == L_with_blanks-1:
                                        betas[s,t] = betas[s,t+1] * y_pred[blank,t]
                                else:
                                        betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * y_pred[blank,t]
                        # same label twice
                        elif s == L_with_blanks-2 or seq[l] == seq[l+1]:
                                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * y_pred[seq[l],t]
                        else:
                                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1])* y_pred[seq[l],t]

                c = np.sum(betas[start:end,t])
                betas[start:end,t] = betas[start:end,t] / c
                llBackward += np.log(c)

        grad = np.zeros(y_pred.shape)
        ab = alphas*betas
        for s in xrange(L):
                # blank
                if s%2 == 0:
                        grad[blank,:] += ab[s,:]
                        ab[s,:] = ab[s,:]/y_pred[blank,:]
                else:
                        grad[seq[(s-1)/2],:] += ab[s,:]
                        ab[s,:] = ab[s,:]/(y_pred[seq[(s-1)/2],:])
        absum = np.sum(ab,axis=0)
        grad = y_pred - grad / (y_pred * absum)
        return grad




