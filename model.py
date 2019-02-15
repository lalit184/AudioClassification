from Parameters import Parameters


class LSTM(nn.Module,Parameters):

	def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
					num_layers=2):
		super(LSTM, self).__init__()
		self.input_dim1 = self.WindowSize
		self.input_dim2 = self.WindowSize//4
		self.input_dim3 = self.WindowSize//16
		self.input_dim4 = self.WindowSize//32
		self.input_dim5 = self.WindowSize//64
		
		self.output_dim1 = self.WindowSize//4
		self.output_dim2 = self.WindowSize//16
		self.output_dim3 = self.WindowSize//32
		self.output_dim4 = self.WindowSize//64
		self.output_dim5 = 10

		self.hidden_dim1 = 30
		self.hidden_dim2 = 30


		
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers

		# Define the LSTM layer
		self.LSTM1 = nn.LSTM(self.input_dim1, self.hidden_dim1, self.num_layers1)
		self.LSTM2 = nn.LSTM(self.input_dim2, self.hidden_dim2, self.num_layers2)
		self.LSTM3 = nn.LSTM(self.input_dim3, self.hidden_dim3, self.num_layers3)
		
		# Define the output layer
		self.linear = nn.Linear(self.hidden_dim, output_dim)

	def init_hidden(self):
		# This is what we'll initialise our hidden state as
		return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
				torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

	def forward(self, input):
		# Forward pass through LSTM layer
		# shape of lstm_out: [input_size, batch_size, hidden_dim]
		# shape of self.hidden: (a, b), where a and b both 
		# have shape (num_layers, batch_size, hidden_dim).
		lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
		
		# Only take the output from the final timetep
		# Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
		y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
		return y_pred.view(-1)

model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)