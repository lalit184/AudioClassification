import os
import glob
import time
import numpy as np
from scipy.io import wavfile
import csv
from Parameters import Parameter
from operator import eq
from scipy import stats
from python_speech_features import mfcc

class DataProcessing(Parameter):
	def __init__(self):
		super(DataProcessing, self).__init__()
		self.Names2Label={	"air_conditioner":0, "car_horn":1,
							"children_playing":2,"dog_bark":3,
							"drilling":4,"engine_idling":5,
							"gun_shot":6,"jackhammer":7,
							"siren":8,"street_music":9
						}
		self.WavFileDirectory='./URBAN-SED_v2.0.0/audio/train/'
		self.TxtAnnotationDirectory='./URBAN-SED_v2.0.0/annotations/train/'
		
	
	def FetchAnnotation(self,Name):
		"""
		We take the annotation data and generate a one hot label for each 
		element in the sound vector.

		For a sound of type 	 [aaaabbbbaaa]
		we have the annotation  [[00001111000],
								 [11110000111]]
		which is a psuedo one hot vector annotationn
		"""
		Annotation=np.zeros(int(self.SignalLength/self.SubSamplingRate))
		with open(Name) as f:
			reader = csv.reader(f, delimiter="\t")
			AnnotationCSV = list(reader)
		
		for category in AnnotationCSV:
			Annotation[	int(self.SamplingFrequency*float(category[0])/self.SubSamplingRate):
						int(self.SamplingFrequency*float(category[1])/self.SubSamplingRate)] =self.Names2Label[category[2]]
		Annotation = Annotation.reshape((-1,int(self.WindowTime*self.SamplingFrequency/self.SubSamplingRate)))
		
		Time2WindowLabel=np.zeros((Annotation.shape[0],self.NumClasses))
		
		for i in range(Annotation.shape[0]):
			Time2WindowLabel[i,:]=np.eye(self.NumClasses)[int(stats.mode(Annotation[i,:]).mode[0])] 
		print("Timed to window",Time2WindowLabel.shape)	
		return Time2WindowLabel

	def FetchSignal(self,Name):
		SamplingFrequency, Data = wavfile.read(Name)
		Data=Data[::self.SubSamplingRate]
		Data=mfcc(signal=Data,samplerate=self.SamplingFrequency/self.SubSamplingRate,nfft=self.WindowSize,nfilt=100,
					winlen=self.WindowTime,winstep=self.WindowStep,winfunc=np.hamming,numcep=self.NumCep)

		"""
		the size of Data is  SignalLength/WindowTime,NumCep
		"""
		return Data

	def FetchInputsAndLabels(self):
		WaveFilesList=[]
		AnnotationFileList=[]

		for file in glob.glob(self.WavFileDirectory+"*.wav"):
			WaveFilesList.append(file)

		for file in glob.glob(self.TxtAnnotationDirectory+"*.txt"):
			AnnotationFileList.append(file)

		WaveFilesList.sort()	
		AnnotationFileList.sort()

		
		for WavFileName,AnnotationFileName in zip(WaveFilesList,AnnotationFileList):
			"""

			Some reshaping op

			"""
			WaveArray=self.FetchSignal(WavFileName)
			LabelArray=self.FetchAnnotation(Name=AnnotationFileName)
			print(WaveArray.shape)
			
			yield WaveArray,LabelArray
						
		


			
				



	
