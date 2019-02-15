class Parameters(object):
	def __init__(self):
		self.NumClasses=10
		self.TotalEpochs=20
		self.SamplingFrequency=44100
		self.SignalLength=441000
		self.WindowTime=0.1
		self.WindowStep=0.1
		self.WindowSize=int(self.WindowTime*self.SamplingFrequency/self.SubSamplingRate)
		self.SubSamplingRate=1
		self.NumCep=13


