from Parameters import Parameter
import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTM(nn.Module):

	def __init__(self):
		super(LSTM, self).__init__()
		self.ParameterObject=Parameter()
		self.input_dim1 = self.ParameterObject.NumCep
		self.input_dim2 = self.ParameterObject.NumCep
		self.input_dim3 = self.ParameterObject.NumCep
		self.input_dim4 = self.ParameterObject.NumCep
		self.input_dim5 = self.ParameterObject.NumCep

		self.InterimOutputDim=100
		self.output_dim5 = 10

		self.num_layers1 = 1
		self.num_layers2 = 1
		self.num_layers3 = 1
		self.num_layers4 = 1
		self.num_layers5 = 1
		
		self.hidden_dim1 = 100
		self.hidden_dim2 = 100
		self.hidden_dim3 = 100
		self.hidden_dim4 = 100
		self.hidden_dim5 = 100

		self.batch_size = 1
		
		
		# Define the LSTM layer
		self.LSTM1 = nn.LSTM(self.input_dim1, self.hidden_dim1, self.num_layers1,batch_first=True)
		self.LSTM2 = nn.LSTM(self.input_dim2, self.hidden_dim2, self.num_layers2,batch_first=True)
		self.LSTM3 = nn.LSTM(self.input_dim3, self.hidden_dim3, self.num_layers3,batch_first=True)
		self.LSTM4 = nn.LSTM(self.input_dim4, self.hidden_dim4, self.num_layers4,batch_first=True)
		self.LSTM5 = nn.LSTM(self.input_dim5, self.hidden_dim5, self.num_layers5,batch_first=True)

		#self.hidden2tag1 = nn.Linear(self.hidden_dim1, self.InterimOutputDim)
		#self.hidden2tag2 = nn.Linear(self.hidden_dim2, self.InterimOutputDim)
		#self.hidden2tag3 = nn.Linear(self.hidden_dim3, self.InterimOutputDim)
		#self.hidden2tag4 = nn.Linear(self.hidden_dim4, self.InterimOutputDim)
		self.hidden2tag = nn.Linear(self.hidden_dim5, self.output_dim5)

		
		

	def init_hidden(self):
		# This is what we'll initialise our hidden state as
		self.hidden1 = (torch.zeros(self.num_layers1, self.ParameterObject.batch_size, self.hidden_dim1),torch.zeros(self.num_layers1, self.batch_size, self.hidden_dim1))
		self.hidden2 = (torch.zeros(self.num_layers2, self.ParameterObject.batch_size, self.hidden_dim2),torch.zeros(self.num_layers2, self.batch_size, self.hidden_dim2))
		self.hidden3 = (torch.zeros(self.num_layers3, self.ParameterObject.batch_size, self.hidden_dim3),torch.zeros(self.num_layers3, self.batch_size, self.hidden_dim3))
		self.hidden4 = (torch.zeros(self.num_layers4, self.ParameterObject.batch_size, self.hidden_dim4),torch.zeros(self.num_layers4, self.batch_size, self.hidden_dim4))
		self.hidden5 = (torch.zeros(self.num_layers5, self.ParameterObject.batch_size, self.hidden_dim5),torch.zeros(self.num_layers5, self.batch_size, self.hidden_dim5))
		return 		

	def forward(self, input):
		# Forward pass through LSTM layer
		# shape of lstm_out: [input_size, batch_size, hidden_dim]
		# shape of self.hidden: (a, b), where a and b both 
		# have shape (num_layers, batch_size, hidden_dim).
		
		lstm_out1, self.hidden1 = self.LSTM1(input.view(-1,100,100),self.hidden1)
		lstm_out1 = F.relu(lstm_out1)
		
		lstm_out2, self.hidden2 = self.LSTM2(lstm_out1.view(-1,100,100),self.hidden2)
		lstm_out2 = F.relu(lstm_out2)		
		
		lstm_out3, self.hidden3 = self.LSTM3(lstm_out2.view(-1,100,100),self.hidden3)
		lstm_out3 = F.relu(lstm_out3)		
		
		lstm_out4, self.hidden4 = self.LSTM4(lstm_out3.view(-1,100,100),self.hidden4)
		lstm_out4 = F.relu(lstm_out4)
		
		lstm_out5, self.hidden5 = self.LSTM5(lstm_out4.view(-1,100,100),self.hidden5)
		
		Output= self.hidden2tag(lstm_out5)
		Output = torch.sigmoid(Output)
		
		
		# Only take the output from the final timetep
		# Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
		
		return Output.view(100,10)

