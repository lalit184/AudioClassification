from Parameters import Parameter
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from DataProcessing import DataProcessing
from model import LSTM
import numpy as np
import time


models = LSTM().double()
models=models.cuda()
loss_function = nn.BCELoss(size_average=True,reduce=True)
optimizer = optim.Adam(models.parameters())

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()

DataObject=DataProcessing()

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
	print("Beginning as a batch")
	StepsOfEpoch=0
	DataMethodObject=DataObject.FetchInputsAndLabels()
	for wav, label in DataMethodObject:
		then=time.time()
		StepsOfEpoch+=1
		# Step 1. Remember that Pytorch accumulates gradients.
		# We need to clear them out before each instance
		models.zero_grad()
		models.init_hidden()
		# Also, we need to clear out the hidden state of the LSTM,
		# detaching it from its history on the last instance.
		output = models(torch.tensor(wav).float())
		#print(output)
		#print(label)
		# Step 4. Compute the loss, gradients, and update the parameters by
		#  calling optimizer.step()
		loss = loss_function(output, torch.tensor(label).float())

		loss.backward()
		optimizer.step()
		now=time.time()
		
		print("Epoch:",epoch,"Step:",StepsOfEpoch," of 2000 steps, Loss:",loss.detach().numpy(),"Time taken for forward and backward is ",now-then)
		